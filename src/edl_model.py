"""
Evidential Dirichlet Deep Learning (D-EDL) Model Architecture
Parameterizes class probabilities via Dirichlet concentration parameters (evidence).
Disentangles aleatoric uncertainty from epistemic uncertainty (vacuity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialNet(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout_rate=0.1):
        super(EvidentialNet, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Backbone feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Latent representation dimension
        self.latent_dim = hidden_dim // 2
        
        # Evidence prediction head (must be non-negative)
        self.evidence_head = nn.Linear(self.latent_dim, num_classes)
        
    def forward_latent(self, x):
        """Extract latent representation for fairness alignment"""
        return self.encoder(x)

    def forward(self, x):
        """
        Returns:
            evidence: non-negative evidence e_k >= 0
            alpha: Dirichlet parameters alpha_k = e_k + 1
            prob: Expected categorical probabilities p_k = alpha_k / S
            uncertainty: Epistemic vacuity u = K / S in (0, 1]
        """
        latent = self.encoder(x)
        # Softplus ensures non-negative evidence
        raw_evidence = self.evidence_head(latent)
        evidence = F.softplus(raw_evidence)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S
        return evidence, alpha, prob, uncertainty

class StandardClassifierNet(nn.Module):
    """Standard Softmax DNN baseline for comparison"""
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout_rate=0.1):
        super(StandardClassifierNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.head = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward_latent(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        latent = self.encoder(x)
        logits = self.head(latent)
        prob = F.softmax(logits, dim=1)
        return logits, prob

def edl_bayes_loss(y, alpha, num_classes=2):
    """
    Bayesian expected loss under the Dirichlet distribution
    y: one-hot or class indices (batch_size,)
    alpha: Dirichlet concentration parameters (batch_size, K)
    """
    if len(y.shape) == 1:
        y_one_hot = (torch.arange(num_classes, device=y.device) == y.unsqueeze(-1)).float()
    else:
        y_one_hot = y.float()
        
    S = torch.sum(alpha, dim=1, keepdim=True)
    # Expected log-loss: sum_k y_k * (digamma(S) - digamma(alpha_k))
    loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    return torch.mean(loss)

def edl_kl_divergence(alpha, y, num_classes=2):
    """
    KL divergence regularizer between non-target evidence Dirichlet and uniform Dirichlet prior Dir(1)
    Penalizes generating evidence for misleading classes.
    """
    if len(y.shape) == 1:
        y_one_hot = (torch.arange(num_classes, device=y.device) == y.unsqueeze(-1)).float()
    else:
        y_one_hot = y.float()
        
    # Remove target evidence
    tilde_alpha = y_one_hot + (1.0 - y_one_hot) * alpha
    
    beta = torch.ones((1, num_classes), dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(tilde_alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(tilde_alpha), dim=1, keepdim=True)
    lnB_prior = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(tilde_alpha)
    dg1 = torch.digamma(S_alpha)
    
    kl = torch.sum((tilde_alpha - beta) * (dg0 - dg1), dim=1, keepdim=True) + lnB + lnB_prior
    return torch.mean(kl)

def edl_total_loss(evidence, alpha, y, num_classes=2, kl_weight=0.1):
    bayes_loss = edl_bayes_loss(y, alpha, num_classes=num_classes)
    kl_loss = edl_kl_divergence(alpha, y, num_classes=num_classes)
    return bayes_loss + kl_weight * kl_loss, bayes_loss, kl_loss
