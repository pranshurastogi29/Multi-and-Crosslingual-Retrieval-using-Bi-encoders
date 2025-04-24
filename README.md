# Multi and Cross-lingual Information Retrieval using Bi-encoders

This project implements a bi-encoder based approach for multi-lingual and cross-lingual information retrieval. It uses transformer-based models to encode queries and documents into a shared semantic space, enabling retrieval across different languages.

## Overview

The system uses a bi-encoder architecture where:

- Queries and documents are encoded into dense vector representations using transformer models
- The encoders are trained to map semantically similar content to similar vector spaces, regardless of language
- Retrieval is performed by computing similarity between query and document vectors
- An ensemble approach combines multiple models for improved performance

## Key Components

### Bi-encoder Training (`bi_encoder_train.ipynb`)

- Implements the bi-encoder model training pipeline
- Uses sentence transformers and transformer models as the base encoders
- Trains the model to optimize similarity between matching query-document pairs
- Supports various training configurations and hyperparameters
- Includes data preprocessing and augmentation

### Ensemble Submission (`ensemble_submission.ipynb`) 

- Implements ensemble approach combining multiple trained models
- Performs retrieval using the trained bi-encoders
- Combines results from different models using a sophisticated ranking algorithm
- Handles submission formatting and evaluation

#### Ensemble Ranking Process

The ensemble combines predictions from multiple models using a sophisticated ranking algorithm:

1. **Model Outputs Processing**
   - Each model generates predictions with document IDs and probability scores
   - Outputs are structured as DataFrames with post IDs, prediction probabilities, and predicted document IDs

2. **Ranking Algorithm (`ensemble_ranks`)**
   - Combines predictions from multiple models
   - Removes duplicate document IDs by keeping the highest probability score
   - Sorts results by probability in descending order
   - Returns top-K (typically top-10) documents for each query

3. **Deduplication Strategy**
   - When the same document is predicted by multiple models, the highest confidence score is retained
   - This ensures that strong predictions from any model are not overlooked

4. **Final Ranking**
   - Results are formatted according to the required submission format
   - Each query is mapped to its top-K most relevant documents
   - The final output is saved in JSON format for evaluation

## Setup

Required packages:
```
pyterrier
unidecode
sentence_transformers
torch_scatter
```

## Usage

1. Train bi-encoder models:
```python
python bi_encoder_train.py
```

2. Run ensemble retrieval:
```python
python ensemble_submission.py
```

## Model Architecture

The bi-encoder architecture consists of:

- Query encoder: Transformer model that encodes queries
- Document encoder: Transformer model that encodes documents  
- Training objective: Maximize similarity between matching query-doc pairs
- Negative sampling: Use hard negatives during training
- Loss functions: Support for various contrastive losses

## Training

The training process:

1. Prepare training data with query-document pairs
2. Encode queries and documents using the bi-encoder
3. Compute loss using positive and negative examples
4. Update model parameters via backpropagation
5. Evaluate on validation set
6. Save best performing models

## Inference

For retrieval:

1. Encode query using query encoder
2. Encode documents using document encoder  
3. Compute similarities between query and document vectors
4. Rank documents by similarity score
5. Combine rankings from multiple models in ensemble

## Prediction Models

The prediction notebooks (`predict-notebook-crosslingual.ipynb` and `predict-notebook-monolingual.ipynb`) use:

- **Base Model**: `intfloat/multilingual-e5-large-instruct`
  - A powerful multilingual transformer model
  - Supports multiple languages for cross-lingual and monolingual retrieval
  - Uses mean pooling for sentence embeddings
  - Batch processing with efficient memory management
  - Configurable sequence lengths (default: 768 tokens)

## Model Weights

To run predictions in your environment:

1. Download our trained model weights from [Kaggle](https://www.kaggle.com/datasets/arkhamking/model-weights/data)
2. The weights are publicly available and include:
   - Monolingual model weights for language-specific retrieval
   - Cross-lingual model weights for cross-language retrieval
   - Different versions trained with various configurations

Place the downloaded weights in a `model-weights` directory in your project root.

## License

MIT License 
