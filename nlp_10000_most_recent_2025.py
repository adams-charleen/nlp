import Bio
from Bio import Entrez
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from datasets import Dataset
import pandas as pd
import numpy as np
import re
import spacy
import time
import logging
import os
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("nlp_run_10000_2025.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Set email for PubMed API
Entrez.email = "cadams9@bidmc.harvard.edu"

# Define search terms with date filter (2025 onwards) and added "BMI" and "body mass index"
def get_search_term():
    """Returns a PubMed search term for cardiometabolic disorders and genomic causation from 2025 onwards."""
    cardiometabolic_terms = (
        "\"Cardiovascular Diseases\"[MeSH Terms] OR \"Coronary Heart Disease\" OR "
        "\"Myocardial Infarction\" OR \"Heart Failure\" OR \"Arrhythmia\" OR "
        "\"Cardiomyopathy\" OR \"Hypertension\"[MeSH Terms] OR \"Hyperlipidemia\" OR "
        "\"Insulin Resistance\" OR \"Metabolic Syndrome\"[MeSH Terms] OR "
        "\"Diabetes Mellitus, Type 2\"[MeSH Terms] OR \"Type 2 Diabetes Mellitus\" OR "
        "\"T2D\" OR \"Non-Insulin-Dependent Diabetes\" OR \"Adult-Onset Diabetes\" OR "
        "\"Diabetic Complications\" OR \"Gestational Diabetes\" OR \"Obesity\"[MeSH Terms] OR "
        "\"Non-alcoholic Fatty Liver Disease\" OR \"Stroke\" OR "
        "\"Peripheral Artery Disease\" OR \"Dyslipidemias\"[MeSH Terms] OR "
        "\"Atherosclerosis\"[MeSH Terms] OR \"BMI\" OR \"body mass index\""
    )
    genetic_terms = (
        "\"Genes\"[MeSH Terms] OR \"Genetic Predisposition to Disease\"[MeSH Terms] OR "
        "\"Mutation\"[MeSH Terms] OR \"Polymorphism, Genetic\"[MeSH Terms] OR "
        "\"Genetic Variant\" OR \"Genome-Wide Association Study\" OR \"GWAS\" OR "
        "\"Single Nucleotide Polymorphism\" OR \"SNP\" OR \"Gene Expression\"[MeSH Terms] OR "
        "\"Transcriptomics\" OR \"Epigenetics\"[MeSH Terms] OR \"Methylation\" OR "
        "\"Pharmacogenomics\" OR \"splicing\" OR \"splice site\""
    )
    causal_terms = (
        "\"Causality\"[MeSH Terms] OR \"Risk Factors\"[MeSH Terms] OR \"Biomarkers\"[MeSH Terms] OR "
        "\"Causes\" OR \"Leads to\" OR \"Is associated with\" OR \"Results in\" OR "
        "\"Contributes to\" OR \"Is linked to\" OR \"Plays a role in\" OR \"causal\""
    )
    date_filter = "(\"2025\"[Date - Publication] : \"3000\"[Date - Publication])"
    return f"({cardiometabolic_terms}) AND ({genetic_terms} OR {causal_terms}) AND {date_filter}"

search_term = get_search_term()

# Fetch 10,000 most recent PubMed abstracts
def fetch_pubmed_data(search_term, max_results=10000, fetch_buffer=11000):
    """Fetches the 10,000 most recent abstracts from PubMed starting 2025."""
    logger.info(f"Fetching up to {max_results} most recent abstracts from 2025 with query: {search_term}")
    try:
        # Search with sorting by publication date (most recent first)
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=fetch_buffer, usehistory="y", sort="pub date")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        webenv = record["WebEnv"]
        query_key = record["QueryKey"]

        articles = []
        batch_size = 100
        for start in range(0, len(id_list), batch_size):
            if len(articles) >= max_results:
                break
            end = min(start + batch_size, len(id_list))
            batch_ids = id_list[start:end]
            retries = 5
            for attempt in range(retries):
                try:
                    handle = Entrez.efetch(
                        db="pubmed", id=",".join(batch_ids), rettype="medline", retmode="text",
                        webenv=webenv, query_key=query_key
                    )
                    batch_text = handle.read()
                    batch_records = batch_text.split("\n\n")
                    for record in batch_records:
                        article = parse_record(record)
                        if article["title"] and article["abstract"]:
                            articles.append(article)
                            if len(articles) % 1000 == 0:
                                logger.info(f"Fetched {len(articles)} articles so far...")
                            if len(articles) >= max_results:
                                break
                    handle.close()
                    break
                except Exception as e:
                    logger.error(f"Error fetching batch {start//batch_size + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        logger.error("Failed after retries. Skipping batch.")
            time.sleep(0.34)  # Rate limit: 3 requests/sec
        logger.info(f"Fetched {len(articles)} articles from 2025 onward.")
        return articles[:max_results]  # Ensure exactly max_results
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return []

# Parse MEDLINE record to extract title, abstract, DOI, journal, date, and authors
def parse_record(record):
    """Parses a MEDLINE record to extract title, abstract, DOI, journal, date, and authors."""
    title = ""
    abstract = ""
    doi = ""
    journal = ""
    date = ""
    authors = []
    current_field = None
    for line in record.split("\n"):
        if line.startswith("TI  - "):
            current_field = "title"
            title = line[6:].strip()
        elif line.startswith("AB  - "):
            current_field = "abstract"
            abstract = line[6:].strip()
        elif line.startswith("LID - ") and "doi" in line.lower():
            parts = line.split()
            for part in parts:
                if "doi" in part.lower():
                    doi = part.replace("[doi]", "").strip()
                    break
        elif line.startswith("TA  - "):
            journal = line[6:].strip()
        elif line.startswith("DP  - "):
            date = line[6:].strip()
        elif line.startswith("AU  - "):
            authors.append(line[6:].strip())
        elif line.startswith(" "):
            if current_field == "title":
                title += " " + line.strip()
            elif current_field == "abstract":
                abstract += " " + line.strip()
        else:
            current_field = None
    authors_str = ", ".join(authors)
    return {"title": title, "abstract": abstract, "doi": doi, "journal": journal, "date": date, "authors": authors_str}

articles = fetch_pubmed_data(search_term)
if not articles:
    logger.warning("No articles fetched (expected in 2024 as data starts from 2025). Proceeding with empty dataset.")

# Load spaCy model for labeling
nlp = spacy.load("en_core_sci_sm")

# Label abstracts with added "bmi" and "body mass index" to cardiometabolic terms
def label_abstracts(articles, batch_size=500):
    """Labels abstracts based on cardiometabolic and causal terms, and calculates rank."""
    cardiometabolic_terms = [
        "cardiovascular disease", "coronary heart disease", "myocardial infarction", "heart failure",
        "arrhythmia", "cardiomyopathy", "hypertension", "hyperlipidemia", "insulin resistance",
        "metabolic syndrome", "type 2 diabetes", "obesity", "non-alcoholic fatty liver disease",
        "stroke", "peripheral artery disease", "dyslipidemia", "atherosclerosis", "bmi", "body mass index"
    ]
    causal_phrases = [
        "causes", "leads to", "is associated with", "results in", "contributes to", "is linked to",
        "plays a role in", "causal"
    ]
    genetic_terms = [
        "gene", "mutation", "polymorphism", "genetic variant", "gwas", "snp", "gene expression",
        "transcriptomics", "epigenetics", "methylation", "pharmacogenomics", "splicing", "splice site"
    ]

    labeled_articles = []
    for batch_start in range(0, len(articles), batch_size):
        batch_end = min(batch_start + batch_size, len(articles))
        batch_articles = articles[batch_start:batch_end]
        logger.info(f"Labeling batch {batch_start//batch_size + 1} (articles {batch_start} to {batch_end})")
        
        for article in batch_articles:
            try:
                abstract_text = article["abstract"].lower()
                found_cardiometabolic = [term for term in cardiometabolic_terms if term in abstract_text]
                found_causal = [phrase for phrase in causal_phrases if phrase in abstract_text]
                found_genetic = [term for term in genetic_terms if term in abstract_text] + re.findall(r'\b[A-Z][A-Z0-9-]*[0-9]+\b', article["abstract"])
                
                # Calculate rank: number of unique terms found
                all_found_terms = found_cardiometabolic + found_causal + found_genetic
                rank = len(set(all_found_terms))  # Unique terms
                
                # Positive (1) if cardiometabolic term + causal/genetic term present
                label = 1 if (len(found_cardiometabolic) > 0 and 
                            (len(found_causal) > 0 or len(found_genetic) > 0)) else 0
                
                labeled_articles.append({
                    "title": article["title"],
                    "journal": article["journal"],
                    "date": article["date"],
                    "authors": article["authors"],
                    "abstract": article["abstract"],
                    "doi": article["doi"],
                    "label": label,
                    "rank": rank,
                    "cardiometabolic_terms": ", ".join(found_cardiometabolic),
                    "causal_terms": ", ".join(found_causal),
                    "genetic_terms": ", ".join(found_genetic)
                })
            except Exception as e:
                logger.error(f"Error labeling article: {e}")
                labeled_articles.append({
                    "title": article["title"],
                    "journal": article["journal"],
                    "date": article["date"],
                    "authors": article["authors"],
                    "abstract": article["abstract"],
                    "doi": article["doi"],
                    "label": 0,
                    "rank": 0,
                    "cardiometabolic_terms": "",
                    "causal_terms": "",
                    "genetic_terms": ""
                })
    return labeled_articles

labeled_articles = label_abstracts(articles)
data = pd.DataFrame(labeled_articles)
label_counts = Counter(data["label"])
logger.info(f"Labeled {len(data)} articles: {label_counts[1]} positive, {label_counts[0]} negative")

# Tokenize for BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["abstract"], padding="max_length", truncation=True, max_length=512)

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()

# Fine-tune BioBERT
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=2).to(device)

# Class weights for imbalanced data
weight_for_0 = (1 / label_counts[0]) * (len(data) / 2.0) if label_counts[0] > 0 else 1.0
weight_for_1 = (1 / label_counts[1]) * (len(data) / 2.0) if label_counts[1] > 0 else 1.0
class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}

# Custom WeightedTrainer with fixed compute_loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./cardiometabolic_biobert_10000_2025/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./cardiometabolic_biobert_10000_2025/logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True if device == "cuda" else False,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save results
output_folder = "./cardiometabolic_biobert_10000_2025"
os.makedirs(output_folder, exist_ok=True)
model.save_pretrained(output_folder)
tokenizer.save_pretrained(output_folder)
data.to_csv(os.path.join(output_folder, "labeled_abstracts.csv"), index=False)

results = trainer.evaluate()
logger.info(f"Evaluation Results: {results}")
logger.info(f"Done! Saved to '{output_folder}'.")
