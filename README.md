# Guided LLM Distiller 🚀

A tool for knowledge distillation of large language models (LLMs) that incorporates both teacher model outputs and original answers as guidance signals. This implementation uses a custom distillation loss to balance learning from both sources.

## 🌟 Key Features

- **Dual-Source Distillation**: Uses both teacher model outputs and original answers to guide the student model
- **Custom KL Divergence Loss**: Specialized loss function that balances learning from both teacher and original outputs
- **CUDA Support**: Optimized for GPU training
- **Batch Processing**: Efficient batch-wise training implementation
- **Progress Tracking**: Detailed epoch and batch-level progress monitoring

## 🛠️ Technical Implementation

The distillation process follows these key steps:

1. Load teacher and student models
2. Process training data into batches
3. Apply custom distillation loss that considers:
   - Teacher model logits
   - Original answer logits
   - Student model predictions
4. Train using AdamW optimizer
5. Save the distilled model

### Custom Distillation Loss
```python
class DistillationLoss(nn.Module):
    """KL Divergence Loss for Knowledge Distillation"""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
```

## 📊 Training Process
The training loop:
1. Processes data in batches
2. Gets logits from teacher model and original answers
3. Computes student model predictions
4. Applies custom loss function
5. Updates model weights
6. Tracks progress with detailed logging

## 🚀 Usage

Required setup:
- CUDA-capable GPU
- PyTorch
- Transformers library
- Training data in appropriate format

## 🔧 Configuration

Key parameters:
- Temperature: 2.0 (for softening probability distributions)
- Learning rate: 5e-5
- Optimizer: AdamW
- Training epochs: 3 (default)
- Example format for training data:

```json
{"messages": [{"role": "system", "content": "You are an expert AI assistant providing clear and concise answers."}, {"role": "user", "content": "What are the key requirements for AI compliance?"}, {"role": "assistant", "content": "The key requirements include data quality standards, risk management frameworks, and transparency measures. Organizations must implement robust governance policies, conduct regular audits, and maintain documentation of AI systems."}]}
{"messages": [{"role": "system", "content": "You are an expert AI assistant providing clear and concise answers."}, {"role": "user", "content": "How should we handle AI transparency?"}, {"role": "assistant", "content": "Organizations should provide clear documentation of AI decision-making processes, maintain audit trails, and ensure users understand how AI systems operate. Regular updates and accessible explanations of AI functionality are essential."}]}  
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.