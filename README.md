# CoreSetSelection

The project supports Matrix Factorization (MF) and Neural Collaborative Filtering (NCF) models, as well as explicit and implicit feedback types. Coreset methods include random, clustering, gradient, and greedy variants (including no-pretrain versions). It also includes an insights analysis module to evaluate coreset quality.

## Features
- **Model Support**: SimpleMF (MF) and NCF.
- **Feedback Types**: Explicit (rating prediction, evaluated with RMSE) and Implicit (interaction prediction, evaluated with Recall@K and NDCG@K).
- **Coreset Methods**: Full data training, Random sampling, Clustering (with/without pretrain), Gradient (with/without pretrain), Greedy (with/without pretrain).
- **Evaluation**: Performance metrics on the test set.
- **Insights**: Optional analysis of coreset interaction value, representativeness, diversity, etc. (generates TXT reports and PNG charts).

## Requirements
The project uses Python 3.8+. Install the following packages using pip:
- **torch**: For model training and inference.
- **pandas**: For data loading and processing.
- **numpy**: For numerical computations.
- **scikit-learn**: For clustering, nearest neighbors, and distance calculations.
- **tqdm**: For progress bars.
- **matplotlib**: For insights visualizations.
- **scipy**: For statistical computations.

Note: The project assumes the MovieLens 1M dataset is downloaded (see below).

## Data Preparation
1. Download the MovieLens 1M dataset from [GroupLens](https://grouplens.org/datasets/movielens/1m/) as `ml-1m.zip`.
2. Extract it and place `ratings.dat` in the `ml-1m/` folder in the project root (path: `ml-1m/ratings.dat`).
3. The data loading script (`data.py`) automatically handles leave-one-out splitting: sorts by timestamp, uses the last interaction as test, randomly selects one from the rest for validation, and the remainder for training.

## How to Run
The entry point is `main.py`, configured via command-line arguments. Ensure the dataset is in place before running.

## Basic command: 
python main.py 

### Key Arguments
- `--methods`: Coreset methods to run, comma-separated (e.g., `full,random`) or `all` (default: `all`). Available: `full`, `random`, `clustering`, `gradient`, `gradient_no_pretrain`, `greedy`, `greedy_no_pretrain`.
- `--batch_size`: Training batch size (default: 256).
- `--epochs`: Number of training epochs (default: 20).
- `--lr`: Learning rate (default: 0.001).
- `--k`: Top-K for evaluation (default: 10).
- `--model_type`: Model type, `NCF` or `MF` (default: `NCF`).
- `--feedback_type`: Feedback type, `explicit` or `implicit` (default: `explicit`).
- `--debug`: Enable debug prints (optional).
- `--insights`: Run insights analysis (optional, generates `insights_*.txt` and PNG charts).

Results are printed to the console and saved to a CSV file (e.g., `results_NCF_explicit.csv`). If `--insights` is enabled, additional insights reports are generated.

### Switching Modes (MF/NCF, Explicit/Implicit). Using NCF+Explicit as an example
Use `--model_type` and `--feedback_type` to switch modes. Example commands:

**NCF + Explicit (Default Mode)**: python main.py --methods=all --epochs=20
   
