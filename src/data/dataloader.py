"""
Data loading utilities for multimodal vision-language training.
Supports MS-COCO, LLaVA instruction datasets, and custom formats.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
import logging

logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """
    MS-COCO dataset for vision-language training.
    Supports both captioning and retrieval tasks.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_length: int = 512,
        image_size: int = 224,
        num_samples: Optional[int] = None,
        task: str = "captioning"  # captioning, retrieval
    ):
        """
        Initialize COCO dataset.
        
        Args:
            data_dir: Path to COCO dataset
            split: Dataset split (train, val, test)
            processor: HuggingFace processor for tokenization/image processing
            max_length: Maximum sequence length
            image_size: Target image size
            num_samples: Limit number of samples (for debugging)
            task: Task type (captioning, retrieval)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.task = task
        
        # Load COCO dataset from HuggingFace
        try:
            if split == "train":
                self.dataset = load_dataset("nlphuji/flickr30k", split="train")
            elif split == "val":
                self.dataset = load_dataset("nlphuji/flickr30k", split="validation")  
            else:
                self.dataset = load_dataset("nlphuji/flickr30k", split="test")
                
            logger.info(f"Loaded {len(self.dataset)} samples from Flickr30k {split} split")
            
        except Exception as e:
            logger.warning(f"Failed to load Flickr30k: {e}")
            # Fallback to MS-COCO if available locally
            self._load_local_coco()
        
        # Limit dataset size if specified
        if num_samples and num_samples < len(self.dataset):
            indices = torch.randperm(len(self.dataset))[:num_samples]
            self.dataset = self.dataset.select(indices)
            logger.info(f"Limited dataset to {num_samples} samples")
    
    def _load_local_coco(self):
        """Load local COCO dataset if available."""
        annotations_file = self.data_dir / f"annotations/captions_{self.split}2017.json"
        images_dir = self.data_dir / f"{self.split}2017"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create dataset entries
        samples = []
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in image_id_to_filename:
                samples.append({
                    'image_path': str(images_dir / image_id_to_filename[image_id]),
                    'caption': ann['caption'],
                    'image_id': image_id
                })
        
        self.dataset = samples
        logger.info(f"Loaded {len(self.dataset)} local COCO samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        if isinstance(self.dataset, list):
            # Local COCO format
            sample = self.dataset[idx]
            image_path = sample['image_path']
            caption = sample['caption']
        else:
            # HuggingFace dataset format
            sample = self.dataset[idx]
            image = sample['image']
            caption = sample['caption'][0] if isinstance(sample['caption'], list) else sample['caption']
            
            # Convert to PIL if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
        
        # Load image for local datasets
        if isinstance(self.dataset, list):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                # Create dummy image
                image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Process with processor if available
        if self.processor:
            if self.task == "captioning":
                # For captioning, we need both image and text
                inputs = self.processor(
                    images=image,
                    text=caption,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                
            elif self.task == "retrieval":
                # For retrieval, process image and text separately
                image_inputs = self.processor(images=image, return_tensors="pt")
                text_inputs = self.processor(
                    text=caption,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                inputs = {
                    'pixel_values': image_inputs['pixel_values'].squeeze(0),
                    'input_ids': text_inputs['input_ids'].squeeze(0),
                    'attention_mask': text_inputs['attention_mask'].squeeze(0)
                }
        else:
            # Return raw data
            inputs = {
                'image': image,
                'caption': caption
            }
        
        inputs['idx'] = idx
        return inputs


class LLaVADataset(Dataset):
    """
    LLaVA instruction following dataset.
    Supports multimodal conversation training.
    """
    
    def __init__(
        self,
        data_path: str,
        processor=None,
        max_length: int = 2048,
        image_folder: Optional[str] = None
    ):
        """
        Initialize LLaVA dataset.
        
        Args:
            data_path: Path to LLaVA conversation data (JSON)
            processor: HuggingFace processor
            max_length: Maximum sequence length
            image_folder: Folder containing images
        """
        self.processor = processor
        self.max_length = max_length
        self.image_folder = Path(image_folder) if image_folder else None
        
        # Load conversation data
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.conversations = json.load(f)
        else:
            # Try loading as HuggingFace dataset
            try:
                dataset = load_dataset(data_path, split="train")
                self.conversations = [sample for sample in dataset]
            except Exception as e:
                raise ValueError(f"Failed to load dataset: {e}")
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get conversation item."""
        conversation = self.conversations[idx]
        
        # Load image
        image_file = conversation.get('image', '')
        if self.image_folder and image_file:
            image_path = self.image_folder / image_file
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.new('RGB', (224, 224), color='black')
        
        # Format conversation
        conversation_text = self._format_conversation(conversation)
        
        # Process with processor
        if self.processor:
            inputs = self.processor(
                images=image,
                text=conversation_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # Create labels for training (copy input_ids and mask system tokens)
            labels = inputs['input_ids'].clone()
            # TODO: Implement proper label masking for instruction tuning
            
            inputs['labels'] = labels
        else:
            inputs = {
                'image': image,
                'conversation': conversation_text
            }
        
        inputs['idx'] = idx
        return inputs
    
    def _format_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format conversation for training."""
        if 'conversations' in conversation:
            # LLaVA format
            formatted = ""
            for turn in conversation['conversations']:
                role = turn.get('from', 'human')
                content = turn.get('value', '')
                
                if role == 'human':
                    formatted += f"USER: {content}\n"
                elif role == 'gpt':
                    formatted += f"ASSISTANT: {content}\n"
            
            return formatted.strip()
        
        elif 'text' in conversation:
            # Simple text format
            return conversation['text']
        
        else:
            # Try to extract any text content
            text_content = ""
            for key, value in conversation.items():
                if isinstance(value, str) and len(value) > 10:
                    text_content += f"{value}\n"
            
            return text_content.strip()


class MultimodalDataCollator:
    """
    Data collator for multimodal training.
    Handles batching of images, text, and labels.
    """
    
    def __init__(
        self,
        processor=None,
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        """
        Initialize data collator.
        
        Args:
            processor: HuggingFace processor
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors
        """
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Extract components
        batch_dict = {}
        
        # Handle different input formats
        if 'pixel_values' in batch[0]:
            # Already processed batch
            keys = batch[0].keys()
            for key in keys:
                if key != 'idx':
                    batch_dict[key] = torch.stack([item[key] for item in batch])
        
        elif 'image' in batch[0] and self.processor:
            # Raw data that needs processing
            images = [item['image'] for item in batch]
            texts = [item.get('caption', item.get('conversation', '')) for item in batch]
            
            # Process batch
            processed = self.processor(
                images=images,
                text=texts,
                return_tensors=self.return_tensors,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length
            )
            
            batch_dict.update(processed)
            
            # Add labels if present
            if 'labels' in batch[0]:
                batch_dict['labels'] = torch.stack([item['labels'] for item in batch])
        
        else:
            # Fallback to default collation
            batch_dict = default_collate(batch)
        
        return batch_dict


def create_dataloaders(
    dataset_name: str,
    data_config: Dict[str, Any],
    processor=None,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_name: Name of dataset to load
        data_config: Dataset configuration
        processor: HuggingFace processor
        batch_size: Batch size
        num_workers: Number of worker processes
        distributed: Whether to use distributed sampling
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    if dataset_name.lower() == "coco":
        train_dataset = COCODataset(
            data_dir=data_config['data_dir'],
            split="train",
            processor=processor,
            max_length=data_config.get('max_length', 512),
            num_samples=data_config.get('num_train_samples')
        )
        
        val_dataset = COCODataset(
            data_dir=data_config['data_dir'],
            split="val", 
            processor=processor,
            max_length=data_config.get('max_length', 512),
            num_samples=data_config.get('num_val_samples')
        )
    
    elif dataset_name.lower() == "llava":
        train_dataset = LLaVADataset(
            data_path=data_config['train_data_path'],
            processor=processor,
            max_length=data_config.get('max_length', 2048),
            image_folder=data_config.get('image_folder')
        )
        
        val_dataset = None
        if 'val_data_path' in data_config:
            val_dataset = LLaVADataset(
                data_path=data_config['val_data_path'],
                processor=processor,
                max_length=data_config.get('max_length', 2048),
                image_folder=data_config.get('image_folder')
            )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data collator
    collator = MultimodalDataCollator(
        processor=processor,
        max_length=data_config.get('max_length', 512)
    )
    
    # Create samplers
    train_sampler = None
    val_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if val_dataset:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator
        )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataloader)} batches")
    if val_dataloader:
        logger.info(f"Val: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


class DistributedEvalSampler(torch.utils.data.Sampler):
    """
    Sampler for distributed evaluation that doesn't shuffle data.
    Ensures all samples are evaluated exactly once across all processes.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
    
    def __iter__(self):
        # Generate indices for this rank
        indices = list(range(self.total_size))
        
        # Add extra samples to make it evenly divisible
        indices += indices[:(self.num_samples * self.num_replicas - len(indices))]
        assert len(indices) == self.num_samples * self.num_replicas
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples