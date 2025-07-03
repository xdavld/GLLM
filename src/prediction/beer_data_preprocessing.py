import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import re

class BeerDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.text_vocab = {}
        self.vocab_size = 0
        
    def load_and_clean_data(self, csv_path):
        """Load and perform initial cleaning of the beer survey data"""
        df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
        
        # Create shorter column names for easier handling
        column_mapping = {
            'Welchem Geschlecht fühlen Sie sich zugehörig?': 'gender',
            'Wie alt sind Sie?': 'age',
            'Wie ist Ihre derzeitige berufliche Situation?': 'job_status',
            'Von welcher Brauerei stammt das Bier?': 'brewery',
            'Wie lautet der Produktname des Bieres?': 'beer_name',
            'Welcher Alkoholgehalt ist auf dem Bier angegeben?': 'alcohol_content',
            'Wie würden Sie das Bier auf der nachfolgenden Farbskala einordnen?': 'color_scale',
            'Wie würden Sie die Festigkeit des Bierschaums einstufen?': 'foam_stability',
            'Wie würden Sie die Spritzigkeit (CO2) des Bieres einstufen?': 'carbonation',
            'Welche der nachfolgend aufgeführten Aromakategorie Geruchsaromen können Sie dem Bier zuordnen?': 'aroma_category',
            'Nennen Sie alle Aromen, die Sie eindeutig herausschmecken können.': 'taste_aromas',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Süß]': 'sweet',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Sauer]': 'sour',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Bitter]': 'bitter',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Salzig]': 'salty',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Fruchtig]': 'fruity',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Künstlich]': 'artificial',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Intensität]': 'intensity',
            'Bitte bewerten Sie das Bier gemäß der nachfolgenden Geschmacksrichtungen auf einer Skala von 1-10. Eine 1 ist sehr schwache Ausprägung und eine 10 eine sehr starke Ausprägung. [Dauer]': 'duration',
            'Wie sehr stimmen Sie der folgenden Aussage zu: Die Geschmacksrichtung (z. B. süß, sauer, etc.) des Bieres gefällt mir sehr gut.': 'taste_satisfaction',
            'Nennen Sie die Geschmacksrichtung, die aus Ihrer Sicht besonders deutlich zum Vorschein getreten ist. Bei mehreren Antwortmöglichkeiten durch Komma separieren. (z.B. süß, sauer, etc.)': 'prominent_taste',
            'Welche Aromen des Bieres gefallen Ihnen am besten? Bei mehreren Antwortmöglichkeiten durch Komma separieren.': 'liked_aromas',
            'Welche Aromen des Bieres gefallen Ihnen am wenigsten? Bei mehreren Antwortmöglichkeiten durch Komma separieren': 'disliked_aromas',
            'Wie sehr stimmen Sie der folgenden Aussage zu: Das Bier schmeckt mir insgesamt sehr gut.': 'overall_satisfaction',
            'Welche Zutat oder welches Aroma könnten Sie sich in diesem Bier vorstellen, um das Geschmackserlebnis zu verbessern? ': 'improvement_suggestions',
            'Wie wahrscheinlich ist es, dass Sie dieses Bier weiterempfehlen würden?': 'recommendation_score'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Clean alcohol content (remove % sign and convert commas to dots)
        df['alcohol_content'] = df['alcohol_content'].astype(str).str.replace('%', '').str.replace(',', '.')
        df['alcohol_content'] = pd.to_numeric(df['alcohol_content'], errors='coerce')
        
        # Remove rows with too many missing values
        df = df.dropna(thresh=len(df.columns) * 0.5)  # Keep rows with at least 50% non-null values
        
        return df
    
    def extract_numeric_features(self, df):
        """Extract and normalize numeric features"""
        numeric_cols = ['alcohol_content', 'color_scale', 'foam_stability', 'carbonation',
                       'sweet', 'sour', 'bitter', 'salty', 'fruity', 'artificial', 
                       'intensity', 'duration', 'recommendation_score']
        
        # Only keep columns that exist in the dataframe
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        print(f"Found numeric columns: {existing_numeric_cols}")
        
        # Fill missing numeric values with median, but handle all-NaN columns
        for col in existing_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            if pd.isna(median_val):  # If all values are NaN, use 0
                print(f"Warning: Column {col} has all NaN values, filling with 0")
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
        
        # Extract numeric data
        if existing_numeric_cols:
            numeric_data = df[existing_numeric_cols].values
            print(f"Numeric data shape before normalization: {numeric_data.shape}")
            print(f"Numeric data sample:\n{numeric_data[:3]}")
            
            # Check for any remaining NaN or inf values
            if np.any(np.isnan(numeric_data)) or np.any(np.isinf(numeric_data)):
                print("Warning: Found NaN or inf values in numeric data, replacing with 0")
                numeric_data = np.nan_to_num(numeric_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize only if we have valid data
            if numeric_data.shape[0] > 1 and np.std(numeric_data) > 1e-10:
                numeric_data = self.scaler.fit_transform(numeric_data)
            else:
                print("Warning: Cannot normalize data, using raw values")
                # Still fit scaler for consistency
                self.scaler.fit(numeric_data)
        else:
            print("Warning: No numeric columns found")
            numeric_data = np.zeros((len(df), 1))  # Create dummy data
            existing_numeric_cols = ['dummy_numeric']
        
        return numeric_data, existing_numeric_cols
    
    def extract_categorical_features(self, df):
        """Extract and encode categorical features"""
        categorical_cols = ['gender', 'age', 'job_status', 'brewery', 'aroma_category', 
                           'taste_satisfaction', 'overall_satisfaction']
        
        # Only keep columns that exist in the dataframe
        existing_cat_cols = [col for col in categorical_cols if col in df.columns]
        print(f"Found categorical columns: {existing_cat_cols}")
        
        categorical_data = []
        cardinalities = []
        
        for col in existing_cat_cols:
            # Fill missing categorical values with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            
            # Clean the data - remove extra spaces, convert to string
            df[col] = df[col].astype(str).str.strip()
            
            # Encode
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col])
            else:
                encoded = self.label_encoders[col].transform(df[col])
            
            categorical_data.append(encoded)
            cardinalities.append(len(self.label_encoders[col].classes_))
            print(f"Column {col}: {len(self.label_encoders[col].classes_)} unique values")
        
        # Stack categorical features
        if categorical_data:
            categorical_data = np.column_stack(categorical_data)
        else:
            print("Warning: No categorical columns found")
            categorical_data = np.zeros((len(df), 1), dtype=int)  # Create dummy data
            cardinalities = [2]  # Dummy cardinality
            existing_cat_cols = ['dummy_categorical']
        
        return categorical_data, cardinalities, existing_cat_cols
    
    def build_text_vocabulary(self, text_series, max_vocab_size=5000):
        """Build vocabulary from text data"""
        # Combine all text and split into words
        all_text = ' '.join(text_series.fillna('').astype(str))
        
        # Simple tokenization (split by spaces and common punctuation)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Select most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, freq in sorted_words[:max_vocab_size-2]]  # -2 for UNK and PAD
        
        # Build vocabulary mapping
        self.text_vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(vocab_words):
            self.text_vocab[word] = i + 2
        
        self.vocab_size = len(self.text_vocab)
        return self.text_vocab
    
    def text_to_indices(self, text_series, max_length=50):
        """Convert text to token indices"""
        def tokenize_text(text):
            if pd.isna(text) or text == '':
                return [0] * max_length  # PAD tokens
            
            words = re.findall(r'\b\w+\b', str(text).lower())
            indices = [self.text_vocab.get(word, 1) for word in words]  # 1 = UNK
            
            # Pad or truncate to max_length
            if len(indices) < max_length:
                indices.extend([0] * (max_length - len(indices)))  # PAD
            else:
                indices = indices[:max_length]
            
            return indices
        
        return np.array([tokenize_text(text) for text in text_series])
    
    def extract_text_features(self, df, max_vocab_size=5000, max_length=50):
        """Extract and process text features"""
        # Combine text columns for vocabulary building
        text_cols = ['taste_aromas', 'prominent_taste', 'liked_aromas', 
                    'disliked_aromas', 'improvement_suggestions']
        
        # Combine all text data
        all_text = df[text_cols].fillna('').apply(lambda x: ' '.join(x), axis=1)
        
        # Build vocabulary
        self.build_text_vocabulary(all_text, max_vocab_size)
        
        # Convert to indices
        text_indices = self.text_to_indices(all_text, max_length)
        
        return text_indices
    
    def extract_target_variable(self, df):
        """Extract target variable for prediction"""
        # Use overall satisfaction as target, but handle different possible values
        if 'overall_satisfaction' in df.columns:
            target = df['overall_satisfaction'].fillna('Neutral')
        elif 'recommendation_score' in df.columns:
            # Use recommendation score as backup
            target = df['recommendation_score'].fillna(5)  # Middle value
            return pd.to_numeric(target, errors='coerce').fillna(5).values
        else:
            print("Warning: No suitable target variable found, using dummy target")
            return np.ones(len(df)) * 3  # Neutral rating
        
        # Print unique values to understand the data
        print(f"Unique target values: {target.unique()}")
        
        # Convert to numeric scale - handle various possible German responses
        satisfaction_mapping = {
            'Stimme überhaupt nicht zu': 1,
            'Stimme nicht zu': 2,
            'Neutral': 3,
            'Stimme zu': 4,
            'Stimme voll und ganz zu': 5,
            # Add English equivalents just in case
            'Strongly disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly agree': 5,
            # Handle any numeric values that might already be there
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5
        }
        
        # Try to map, then convert to numeric if mapping fails
        target_numeric = target.map(satisfaction_mapping)
        
        # If mapping didn't work, try direct numeric conversion
        if target_numeric.isna().all():
            target_numeric = pd.to_numeric(target, errors='coerce')
        
        # Fill any remaining NaN with neutral value
        target_numeric = target_numeric.fillna(3)
        
        print(f"Target distribution: {target_numeric.value_counts().sort_index()}")
        
        return target_numeric.values
    
    def create_condition_vector(self, df):
        """Create condition vector (e.g., age groups)"""
        # Use age as condition
        if 'age' not in df.columns:
            print("Warning: Age column not found, creating dummy conditions")
            # Create dummy age conditions
            condition_vectors = torch.zeros(len(df), 6)
            condition_vectors[:, 0] = 1  # All in first age group
            return condition_vectors
            
        age_mapping = {
            '16-25': 0,
            '26-35': 1,
            '36-45': 2,
            '46-54': 3,
            '55-65': 4,
            '65+': 5,
            # Handle potential variations
            '16-25': 0,
            '25-35': 1,
            '35-45': 2,
            '45-54': 3,
            '54-65': 4,
            '65+': 5
        }
        
        print(f"Unique age values: {df['age'].unique()}")
        
        age_encoded = df['age'].map(age_mapping).fillna(0)  # Default to youngest group
        
        # Convert to one-hot
        condition_vectors = F.one_hot(torch.tensor(age_encoded.values, dtype=torch.long), 
                                    num_classes=6).float()
        
        return condition_vectors
    
    def prepare_training_data(self, csv_path):
        """Complete data preparation pipeline"""
        print("Loading and cleaning data...")
        df = self.load_and_clean_data(csv_path)
        print(f"Loaded {len(df)} samples")
        
        print("Extracting numeric features...")
        x_numeric, numeric_cols = self.extract_numeric_features(df)
        
        print("Extracting categorical features...")
        x_categorical, cardinalities, cat_cols = self.extract_categorical_features(df)
        
        print("Extracting text features...")
        x_text = self.extract_text_features(df)
        
        print("Creating condition vectors...")
        conditions = self.create_condition_vector(df)
        
        print("Extracting target variable...")
        targets = self.extract_target_variable(df)
        
        # Convert to tensors
        x_numeric = torch.tensor(x_numeric, dtype=torch.float32)
        x_categorical = torch.tensor(x_categorical, dtype=torch.long)
        x_text = torch.tensor(x_text, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.float32)
        
        print(f"Data shapes:")
        print(f"  Numeric: {x_numeric.shape}")
        print(f"  Categorical: {x_categorical.shape}")
        print(f"  Text: {x_text.shape}")
        print(f"  Conditions: {conditions.shape}")
        print(f"  Targets: {targets.shape}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Categorical cardinalities: {cardinalities}")
        
        # save feature-name list on the processor for inference
        self.numeric_cols = numeric_cols
        self.categorical_cols = cat_cols

        return {
            'x_numeric': x_numeric,
            'x_categorical': x_categorical,
            'x_text': x_text,
            'conditions': conditions,
            'targets': targets,
            'cardinalities': cardinalities,
            'vocab_size': self.vocab_size,
            'numeric_cols': numeric_cols,
            'categorical_cols': cat_cols
        }

# Example usage:
if __name__ == '__main__':
    processor = BeerDataProcessor()
    
    # Process your data
    data = processor.prepare_training_data('your_beer_survey.csv')
    
    # Split into train/test
    indices = torch.randperm(len(data['targets']))
    train_size = int(0.8 * len(indices))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = {
        'x_numeric': data['x_numeric'][train_indices],
        'x_categorical': data['x_categorical'][train_indices],
        'x_text': data['x_text'][train_indices],
        'conditions': data['conditions'][train_indices],
        'targets': data['targets'][train_indices]
    }
    
    test_data = {
        'x_numeric': data['x_numeric'][test_indices],
        'x_categorical': data['x_categorical'][test_indices],
        'x_text': data['x_text'][test_indices],
        'conditions': data['conditions'][test_indices],
        'targets': data['targets'][test_indices]
    }
    
    print(f"Training samples: {len(train_data['targets'])}")
    print(f"Test samples: {len(test_data['targets'])}")