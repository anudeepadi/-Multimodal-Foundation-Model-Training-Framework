"""
Comprehensive evaluation metrics for vision-language models.
Includes BLEU, CIDEr, CLIP-Score, and retrieval metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import math
from collections import Counter, defaultdict
import re
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class VisionLanguageMetrics:
    """
    Comprehensive metrics collection for vision-language tasks.
    Supports captioning, retrieval, and VQA evaluation.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize metrics collection.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.bleu_scorer = BLEUScore()
        self.cider_scorer = CIDErScore()
        self.clip_scorer = None  # Initialize lazily
        
    def compute_captioning_metrics(
        self,
        predictions: List[str],
        references: List[List[str]],
        images: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Compute captioning metrics.
        
        Args:
            predictions: Generated captions
            references: Reference captions (list of lists)
            images: Optional images for CLIP-Score
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # BLEU scores
        bleu_scores = self.bleu_scorer.compute(predictions, references)
        metrics.update(bleu_scores)
        
        # CIDEr score
        cider_score = self.cider_scorer.compute(predictions, references)
        metrics['cider'] = cider_score
        
        # ROUGE-L score
        rouge_l = self._compute_rouge_l(predictions, references)
        metrics['rouge_l'] = rouge_l
        
        # METEOR score (if available)
        try:
            meteor_score = self._compute_meteor(predictions, references)
            metrics['meteor'] = meteor_score
        except ImportError:
            logger.warning("METEOR not available (requires nltk.translate.meteor_score)")
        
        # CLIP-Score (if images provided)
        if images is not None:
            if self.clip_scorer is None:
                self.clip_scorer = CLIPScore(device=self.device)
            
            clip_score = self.clip_scorer.compute(predictions, images)
            metrics['clip_score'] = clip_score
        
        return metrics
    
    def compute_retrieval_metrics(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            query_embeddings: Query embeddings [N, D]
            key_embeddings: Key embeddings [M, D]  
            labels: Ground truth labels [N] (if available)
            k_values: Values of k for recall@k computation
            
        Returns:
            Dictionary of retrieval metrics
        """
        # Compute similarity matrix
        similarities = torch.matmul(query_embeddings, key_embeddings.T)
        
        if labels is None:
            # Assume diagonal matching (query i matches key i)
            labels = torch.arange(len(query_embeddings))
        
        metrics = {}
        
        # Recall@K
        for k in k_values:
            recall_k = self._compute_recall_at_k(similarities, labels, k)
            metrics[f'recall_at_{k}'] = recall_k
        
        # Mean Reciprocal Rank
        mrr = self._compute_mrr(similarities, labels)
        metrics['mrr'] = mrr
        
        # Mean Average Precision
        map_score = self._compute_map(similarities, labels)
        metrics['map'] = map_score
        
        # Median Rank
        median_rank = self._compute_median_rank(similarities, labels)
        metrics['median_rank'] = median_rank
        
        return metrics
    
    def compute_vqa_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Compute VQA accuracy metrics.
        
        Args:
            predictions: Predicted answers
            ground_truth: Ground truth answers
            
        Returns:
            Dictionary of VQA metrics
        """
        # Normalize answers
        pred_normalized = [self._normalize_answer(pred) for pred in predictions]
        gt_normalized = [self._normalize_answer(gt) for gt in ground_truth]
        
        # Exact match accuracy
        exact_match = sum(p == g for p, g in zip(pred_normalized, gt_normalized)) / len(predictions)
        
        # Token-level F1
        f1_scores = []
        for pred, gt in zip(pred_normalized, gt_normalized):
            f1_scores.append(self._compute_f1(pred, gt))
        
        avg_f1 = np.mean(f1_scores)
        
        return {
            'vqa_accuracy': exact_match,
            'vqa_f1': avg_f1
        }
    
    def _compute_recall_at_k(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        k: int
    ) -> float:
        """Compute Recall@K."""
        _, top_k_indices = torch.topk(similarities, k, dim=1)
        
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_indices[i]:
                correct += 1
        
        return correct / len(labels)
    
    def _compute_mrr(self, similarities: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute Mean Reciprocal Rank."""
        _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        
        reciprocal_ranks = []
        for i, label in enumerate(labels):
            rank = (sorted_indices[i] == label).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank[0].item() + 1))
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def _compute_map(self, similarities: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute Mean Average Precision."""
        _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        
        average_precisions = []
        for i, label in enumerate(labels):
            # Find position of correct item
            correct_positions = (sorted_indices[i] == label).nonzero(as_tuple=True)[0]
            
            if len(correct_positions) > 0:
                pos = correct_positions[0].item() + 1
                average_precisions.append(1.0 / pos)
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions)
    
    def _compute_median_rank(self, similarities: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute median rank."""
        _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        
        ranks = []
        for i, label in enumerate(labels):
            rank = (sorted_indices[i] == label).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks.append(rank[0].item() + 1)
            else:
                ranks.append(len(sorted_indices[i]))  # Worst possible rank
        
        return np.median(ranks)
    
    def _compute_rouge_l(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """Compute ROUGE-L score."""
        scores = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            
            max_score = 0
            for ref in refs:
                ref_tokens = word_tokenize(ref.lower())
                lcs_length = self._longest_common_subsequence(pred_tokens, ref_tokens)
                
                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    score = 0
                else:
                    precision = lcs_length / len(pred_tokens)
                    recall = lcs_length / len(ref_tokens)
                    
                    if precision + recall == 0:
                        score = 0
                    else:
                        score = 2 * precision * recall / (precision + recall)
                
                max_score = max(max_score, score)
            
            scores.append(max_score)
        
        return np.mean(scores)
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _compute_meteor(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """Compute METEOR score."""
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for pred, refs in zip(predictions, references):
            # METEOR expects a single reference string, so we take the first one
            ref = refs[0] if refs else ""
            score = meteor_score([ref], pred)
            scores.append(score)
        
        return np.mean(scores)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize VQA answer."""
        # Convert to lowercase and remove punctuation
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = prediction.split()
        gt_tokens = ground_truth.split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall)


class BLEUScore:
    """BLEU score computation for caption evaluation."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def compute(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
        bleu_scores = {f'bleu_{i}': [] for i in range(1, 5)}
        
        for pred, refs in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower()) for ref in refs]
            
            for n in range(1, 5):
                weights = [1.0/n] * n + [0] * (4-n)
                try:
                    score = sentence_bleu(
                        ref_tokens,
                        pred_tokens,
                        weights=weights,
                        smoothing_function=self.smoothing.method1
                    )
                    bleu_scores[f'bleu_{n}'].append(score)
                except ZeroDivisionError:
                    bleu_scores[f'bleu_{n}'].append(0.0)
        
        return {k: np.mean(v) for k, v in bleu_scores.items()}


class CIDErScore:
    """CIDEr score computation for caption evaluation."""
    
    def __init__(self):
        self.n_gram = 4
        self.sigma = 6.0
    
    def compute(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> float:
        """Compute CIDEr score."""
        # Compute document frequencies
        all_refs = []
        for refs in references:
            all_refs.extend(refs)
        
        doc_frequencies = self._compute_doc_frequencies(all_refs)
        
        # Compute CIDEr for each prediction
        cider_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_ngrams = self._extract_ngrams(pred)
            
            ref_scores = []
            for ref in refs:
                ref_ngrams = self._extract_ngrams(ref)
                score = self._compute_cider_single(pred_ngrams, ref_ngrams, doc_frequencies)
                ref_scores.append(score)
            
            # Average across references
            avg_score = np.mean(ref_scores) if ref_scores else 0.0
            cider_scores.append(avg_score)
        
        return np.mean(cider_scores)
    
    def _extract_ngrams(self, text: str) -> Dict[str, int]:
        """Extract n-grams from text."""
        tokens = word_tokenize(text.lower())
        ngrams = defaultdict(int)
        
        for n in range(1, self.n_gram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams[ngram] += 1
        
        return ngrams
    
    def _compute_doc_frequencies(self, documents: List[str]) -> Dict[str, int]:
        """Compute document frequencies for all n-grams."""
        doc_freq = defaultdict(int)
        
        for doc in documents:
            ngrams = set(self._extract_ngrams(doc).keys())
            for ngram in ngrams:
                doc_freq[ngram] += 1
        
        return doc_freq
    
    def _compute_cider_single(
        self,
        pred_ngrams: Dict[str, int],
        ref_ngrams: Dict[str, int],
        doc_frequencies: Dict[str, int]
    ) -> float:
        """Compute CIDEr score for a single prediction-reference pair."""
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        # Compute TF-IDF vectors
        pred_tfidf = self._compute_tfidf(pred_ngrams, doc_frequencies)
        ref_tfidf = self._compute_tfidf(ref_ngrams, doc_frequencies)
        
        # Compute cosine similarity
        dot_product = sum(pred_tfidf[k] * ref_tfidf[k] for k in pred_tfidf if k in ref_tfidf)
        
        pred_norm = math.sqrt(sum(v**2 for v in pred_tfidf.values()))
        ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
        
        if pred_norm == 0 or ref_norm == 0:
            return 0.0
        
        cosine_sim = dot_product / (pred_norm * ref_norm)
        
        # Apply CIDEr penalty for length difference
        penalty = math.exp(-abs(len(pred_ngrams) - len(ref_ngrams)) / self.sigma)
        
        return cosine_sim * penalty
    
    def _compute_tfidf(
        self,
        ngrams: Dict[str, int],
        doc_frequencies: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute TF-IDF weights."""
        tfidf = {}
        total_ngrams = sum(ngrams.values())
        total_docs = sum(doc_frequencies.values())
        
        for ngram, count in ngrams.items():
            tf = count / total_ngrams
            df = doc_frequencies.get(ngram, 1)
            idf = math.log(total_docs / df)
            tfidf[ngram] = tf * idf
        
        return tfidf


class CLIPScore:
    """CLIP-Score for semantic similarity between images and captions."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: CLIP model to use
            device: Device to run on
        """
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = device
        except ImportError:
            raise ImportError("CLIP not available. Install with: pip install transformers")
    
    def compute(self, captions: List[str], images: List) -> float:
        """
        Compute CLIP-Score between captions and images.
        
        Args:
            captions: List of captions
            images: List of PIL Images
            
        Returns:
            Average CLIP-Score
        """
        scores = []
        
        with torch.no_grad():
            for caption, image in zip(captions, images):
                # Process inputs
                inputs = self.processor(
                    text=[caption],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Get features
                outputs = self.model(**inputs)
                
                # Compute cosine similarity
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarity
                similarity = torch.sum(image_features * text_features, dim=-1)
                scores.append(similarity.item())
        
        return np.mean(scores)


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Convenience function for computing retrieval metrics."""
    metrics_computer = VisionLanguageMetrics()
    return metrics_computer.compute_retrieval_metrics(query_embeddings, key_embeddings, labels, k_values)


def compute_captioning_metrics(
    predictions: List[str],
    references: List[List[str]],
    images: Optional[List] = None
) -> Dict[str, float]:
    """Convenience function for computing captioning metrics."""
    metrics_computer = VisionLanguageMetrics()
    return metrics_computer.compute_captioning_metrics(predictions, references, images)