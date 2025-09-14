"""
Data preprocessing utilities for multimodal vision-language models.
Handles image augmentation, text tokenization, and format conversion.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import random
import math
import cv2
from transformers import AutoTokenizer, AutoProcessor
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Advanced image preprocessing with augmentation strategies.
    Supports various augmentation techniques for robust training.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = True,
        augment_prob: float = 0.5
    ):
        """
        Initialize image preprocessor.
        
        Args:
            image_size: Target image size
            mean: Normalization mean values
            std: Normalization std values  
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying each augmentation
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Augmentation transforms
        self.augment_transforms = self._create_augment_transforms()
    
    def _create_augment_transforms(self) -> transforms.Compose:
        """Create augmentation pipeline."""
        augmentations = []
        
        # Geometric augmentations
        augmentations.extend([
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            )
        ])
        
        # Color augmentations  
        augmentations.extend([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            RandomGaussianBlur(p=0.3),
            RandomSolarization(p=0.2)
        ])
        
        # Final normalization
        augmentations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(augmentations)
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Process single image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.augment and random.random() < self.augment_prob:
            return self.augment_transforms(image)
        else:
            return self.base_transform(image)
    
    def process_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
        """Process batch of images."""
        processed = [self(image) for image in images]
        return torch.stack(processed)


class RandomGaussianBlur:
    """Random Gaussian blur augmentation."""
    
    def __init__(self, p: float = 0.5, kernel_size: int = 5, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image


class RandomSolarization:
    """Random solarization augmentation."""
    
    def __init__(self, p: float = 0.5, threshold: int = 128):
        self.p = p
        self.threshold = threshold
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return F.solarize(image, threshold=self.threshold)
        return image


class TextPreprocessor:
    """
    Text preprocessing with advanced tokenization strategies.
    Handles various text formats and special tokens.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True
    ):
        """
        Initialize text preprocessor.
        
        Args:
            tokenizer_name: HuggingFace tokenizer identifier
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        
        # Add special tokens if needed
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "cls_token": "[CLS]"
        }
        
        # Only add tokens that don't exist
        tokens_to_add = {}
        for key, token in special_tokens.items():
            if getattr(self.tokenizer, key, None) is None:
                tokens_to_add[key] = token
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens(tokens_to_add)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of tokens."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def prepare_instruction_data(
        self,
        conversations: List[Dict[str, Any]],
        system_message: str = "You are a helpful assistant."
    ) -> List[str]:
        """
        Prepare instruction tuning data with proper formatting.
        
        Args:
            conversations: List of conversation dictionaries
            system_message: System prompt to prepend
            
        Returns:
            List of formatted conversation strings
        """
        formatted_conversations = []
        
        for conv in conversations:
            formatted = system_message + "\n\n"
            
            if 'conversations' in conv:
                # LLaVA-style format
                for turn in conv['conversations']:
                    role = turn.get('from', 'human')
                    content = turn.get('value', '')
                    
                    if role == 'human':
                        formatted += f"Human: {content}\n\n"
                    elif role == 'gpt':
                        formatted += f"Assistant: {content}\n\n"
            
            elif 'instruction' in conv:
                # Alpaca-style format
                instruction = conv['instruction']
                input_text = conv.get('input', '')
                output = conv.get('output', '')
                
                if input_text:
                    formatted += f"Human: {instruction}\n{input_text}\n\n"
                else:
                    formatted += f"Human: {instruction}\n\n"
                
                formatted += f"Assistant: {output}\n\n"
            
            formatted_conversations.append(formatted.strip())
        
        return formatted_conversations


class MultimodalPreprocessor:
    """
    Combined image and text preprocessing for multimodal models.
    Coordinates preprocessing across modalities.
    """
    
    def __init__(
        self,
        processor_name: str = "openai/clip-vit-base-patch32",
        image_size: int = 224,
        max_text_length: int = 512,
        augment_images: bool = True
    ):
        """
        Initialize multimodal preprocessor.
        
        Args:
            processor_name: HuggingFace processor identifier
            image_size: Target image size
            max_text_length: Maximum text sequence length
            augment_images: Whether to augment images during training
        """
        self.processor_name = processor_name
        
        # Try to load processor, fallback to separate components
        try:
            self.processor = AutoProcessor.from_pretrained(processor_name)
            self.has_processor = True
        except Exception as e:
            logger.warning(f"Failed to load processor {processor_name}: {e}")
            self.has_processor = False
            
            # Create separate components
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                augment=augment_images
            )
            self.text_preprocessor = TextPreprocessor(
                tokenizer_name=processor_name,
                max_length=max_text_length
            )
    
    def __call__(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process images and text together.
        
        Args:
            images: PIL Images or list of images
            text: Text string or list of strings
            return_tensors: Format for returned tensors
            **kwargs: Additional processor arguments
            
        Returns:
            Dictionary containing processed inputs
        """
        if self.has_processor:
            # Use unified processor
            return self.processor(
                images=images,
                text=text,
                return_tensors=return_tensors,
                **kwargs
            )
        else:
            # Use separate components
            result = {}
            
            if images is not None:
                if isinstance(images, Image.Image):
                    images = [images]
                pixel_values = self.image_preprocessor.process_batch(images)
                result['pixel_values'] = pixel_values
            
            if text is not None:
                text_inputs = self.text_preprocessor(text, return_tensors=return_tensors)
                result.update(text_inputs)
            
            return result
    
    def prepare_training_batch(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training with proper formatting.
        
        Args:
            batch_data: List of data samples
            
        Returns:
            Processed batch dictionary
        """
        images = []
        texts = []
        labels = []
        
        for sample in batch_data:
            if 'image' in sample:
                images.append(sample['image'])
            
            if 'text' in sample:
                texts.append(sample['text'])
            elif 'caption' in sample:
                texts.append(sample['caption'])
            
            if 'label' in sample:
                labels.append(sample['label'])
        
        # Process inputs
        inputs = self(
            images=images if images else None,
            text=texts if texts else None,
            padding=True,
            truncation=True
        )
        
        # Add labels if present
        if labels:
            inputs['labels'] = torch.tensor(labels)
        
        return inputs
    
    def create_contrastive_batch(
        self,
        images: List[Image.Image],
        captions: List[str],
        negative_sampling: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Create batch for contrastive learning (e.g., CLIP training).
        
        Args:
            images: List of images
            captions: List of corresponding captions
            negative_sampling: Whether to include negative samples
            
        Returns:
            Batch formatted for contrastive learning
        """
        batch_size = len(images)
        
        if negative_sampling:
            # Shuffle captions to create negative pairs
            shuffled_indices = torch.randperm(batch_size)
            negative_captions = [captions[i] for i in shuffled_indices]
            
            # Combine positive and negative pairs
            all_images = images + images
            all_captions = captions + negative_captions
            
            # Create labels (first half are positive pairs)
            labels = torch.cat([
                torch.ones(batch_size),
                torch.zeros(batch_size)
            ])
        else:
            all_images = images
            all_captions = captions
            labels = torch.arange(batch_size)  # Diagonal matching
        
        # Process batch
        inputs = self(images=all_images, text=all_captions)
        inputs['labels'] = labels
        
        return inputs


class DataAugmentationPipeline:
    """
    Comprehensive data augmentation pipeline for multimodal training.
    """
    
    def __init__(
        self,
        image_augmentations: Optional[Dict[str, Any]] = None,
        text_augmentations: Optional[Dict[str, Any]] = None,
        cross_modal_augmentations: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_augmentations: Image augmentation configuration
            text_augmentations: Text augmentation configuration  
            cross_modal_augmentations: Cross-modal augmentation configuration
        """
        self.image_augs = image_augmentations or {}
        self.text_augs = text_augmentations or {}
        self.cross_modal_augs = cross_modal_augmentations or {}
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Apply image augmentations."""
        if 'cutout' in self.image_augs:
            image = self._apply_cutout(image, **self.image_augs['cutout'])
        
        if 'mixup' in self.image_augs:
            # Mixup requires batch processing
            pass
        
        return image
    
    def augment_text(self, text: str) -> str:
        """Apply text augmentations."""
        if 'synonym_replacement' in self.text_augs:
            text = self._synonym_replacement(text, **self.text_augs['synonym_replacement'])
        
        if 'random_deletion' in self.text_augs:
            text = self._random_deletion(text, **self.text_augs['random_deletion'])
        
        return text
    
    def _apply_cutout(
        self,
        image: Image.Image,
        size: int = 16,
        p: float = 0.5
    ) -> Image.Image:
        """Apply cutout augmentation."""
        if random.random() > p:
            return image
        
        w, h = image.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        
        # Create mask
        image_array = np.array(image)
        image_array[y:y+size, x:x+size] = 0
        
        return Image.fromarray(image_array)
    
    def _synonym_replacement(
        self,
        text: str,
        num_replacements: int = 1,
        p: float = 0.1
    ) -> str:
        """Replace words with synonyms."""
        words = text.split()
        num_words = len(words)
        
        if num_words == 0:
            return text
        
        # Simple word replacement (could be enhanced with WordNet)
        replacements = {
            'good': ['great', 'excellent', 'wonderful'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'miniature']
        }
        
        for _ in range(min(num_replacements, num_words)):
            if random.random() < p:
                idx = random.randint(0, num_words - 1)
                word = words[idx].lower()
                
                if word in replacements:
                    words[idx] = random.choice(replacements[word])
        
        return ' '.join(words)
    
    def _random_deletion(
        self,
        text: str,
        p: float = 0.1
    ) -> str:
        """Randomly delete words."""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        # Randomly delete words
        words = [word for word in words if random.random() > p]
        
        # Ensure at least one word remains
        if len(words) == 0:
            return text
        
        return ' '.join(words)