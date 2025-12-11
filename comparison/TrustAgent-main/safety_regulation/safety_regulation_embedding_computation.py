import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
from os.path import join
import openai
import os


def embedding_computation_contriever(tokenizer, retriever, dire, to_save):
    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        regulations = safety_regulations[d]

        # Apply tokenizer
        inputs = tokenizer(
            regulations, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        outputs = retriever(**inputs)

        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains


def embedding_computation_mistral(tokenizer, retriever, dire, to_save):
    def last_token_pool(last_hidden_states,attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        regulations = safety_regulations[d]

        # Apply tokenizer
        max_length = 4096
        # Tokenize the input texts
        batch_dict = tokenizer(regulations, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        # Compute token embeddings
        outputs = retriever(**batch_dict)

        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains



def get_openai_embedding(text, model="text-embedding-ada-002", use_local=True, local_model_path=None):
    """
    Get text embedding using either local model or API
    Args:
        text: input text
        model: model name (for API) or path (for local)
        use_local: whether to use local model
        local_model_path: path to local model
    """
    text = text.replace("\n", " ")
    
    if use_local:
        # Use local embedding model
        if local_model_path is None:
            local_model_path = "/data/Content_Moderation/BAAI-bge-m3"
        
        # Load local model and tokenizer (cache them for efficiency)
        if not hasattr(get_openai_embedding, 'local_tokenizer'):
            print(f"Loading local embedding model from {local_model_path}...")
            get_openai_embedding.local_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            get_openai_embedding.local_model = AutoModel.from_pretrained(local_model_path)
            get_openai_embedding.local_model.eval()
            if torch.cuda.is_available():
                get_openai_embedding.local_model = get_openai_embedding.local_model.cuda()
            print("Local embedding model loaded successfully!")
        
        # Tokenize and get embeddings
        inputs = get_openai_embedding.local_tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = get_openai_embedding.local_model(**inputs)
            # Use mean pooling for embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            return embeddings[0].numpy().tolist()
    else:
        # Use API (DeepSeek or OpenAI)
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_API_BASE")
        
        try:
            # Try new OpenAI client format (v1.0+)
            from openai import OpenAI
            client_kwargs = {}
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            
            client = OpenAI(**client_kwargs)
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except (ImportError, AttributeError):
            # Fallback to old OpenAI API format (v0.28.x)
            if api_key:
                openai.api_key = api_key
            if base_url:
                openai.api_base = base_url
            return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def embedding_computation_openai(dire, to_save, use_local=True, local_model_path=None):
    """
    Compute embeddings using either local model or OpenAI API
    Args:
        dire: path to safety regulations JSON file
        to_save: directory to save embeddings
        use_local: whether to use local model (default: True)
        local_model_path: path to local embedding model
    """
    with open(dire, "r") as f:
        safety_regulations = json.load(f)

    domains = list(safety_regulations.keys())

    for d in tqdm(domains):
        embeddings = []
        regulations = safety_regulations[d]
        for regulation in regulations:
            embedding = get_openai_embedding(regulation, use_local=use_local, local_model_path=local_model_path)
            embeddings.append(torch.tensor(embedding).unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)

        embedding_directory = join(
            to_save, "safety_regulation_{}_embedding.pt".format(d)
        )

        torch.save(embeddings, embedding_directory)

    return domains
