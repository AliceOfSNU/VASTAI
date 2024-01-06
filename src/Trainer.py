import torch

## model
from transformers import get_cosine_schedule_with_warmup
from transformers import Blip2ForConditionalGeneration
from transformers import Blip2Processor
from transformers import Blip2Config
from transformers import set_seed

## training
import peft
from accelerate import find_executable_batch_size
from accelerate import DistributedType
from accelerate import Accelerator

## training basics
from torch.optim import AdamW
import torch

## Data
import pandas as pd
from Dataset import QualDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

def train(model, dataloader, args):
    accelerator = Accelerator() 
    
    @find_executable_batch_size(starting_batch_size = args["batch_size"])
    def inner_training_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()
        
        optimizer = AdamW(model.parameters(), lr = args["lr"])
        # Instantiate the learning rate scheduler    
        accelerator.print("Instantiate the learning rate scheduler.")
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, 
                                                    num_warmup_steps = 100, 
                                                    num_training_steps = (len(train_loader) * args['num_epochs']) // args['gradient_accumulation_steps'],
                                                    )
        
        model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader['train'], dataloader['val'], lr_scheduler)
        for epoch in args["num_epochs"]:
            print(f"epoch {epoch + 1}/{args['num_epochs']}")
            epoch_loss = 0; n_steps = len(train_loader)
            for step, (image, answer) in enumerate(train_loader):
                output = model(input_ids = answer, 
                               pixel_values = image,
                               labels = answer,
                            )
                del [image, answer]
                
                loss = output.loss 
                loss = loss / args["gradient_accumulation_steps"]
                epoch_loss += loss.item()
                accelerator.backward(loss)
                if step % args["gradient_accumulation_steps"] == 0:
                    optimizer.zero_grad()
                    optimizer.step()
                    lr_scheduler.step()
            print(f"end epoch Loss/train: {epoch_loss/n_steps}")
        
        inner_training_loop()   
        
        
        
def main():
    """pipeline"""
    
    # setup training arguments
    args={
        "pre_process_checkpoint": "Salesforce/blip2-opt-2.7b",
        "checkpoint": "Salesforce/blip2-opt-2.7b",
        "padding" : "longest",
        "max_length" : 16,
        #"mixed_precision" : args.mixed_precision,#"fp16",
        "image_resize" : (256,256),
        "lr" : 2e-2,
        "num_epochs" : 2,
        "batch_size" : 128,
        "gradient_accumulation_steps" : 3, #3,
        "train_dataset": "../data/train",
        "val_dataset": "../data/val"
    }
    
    #load model
    print(f"loading blip model.. chkpt: {args['checkpoint']}")
    preprocessor = Blip2Processor.from_pretrained(args["pre_process_checkpoint"], torch_dtype = torch.float16)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args["checkpoint"], 
        device_map = "auto", 
        torch_dtype = "auto",
        offload_folder = "offload_model", 
        offload_state_dict = True,
    )
    
    # dataset and loaders
    df = {
        "train": pd.read_csv('train.csv'),
        "val": pd.read_csv('test.csv')
    }
    transform = transforms.Compose([
        transforms.Resize(args['image_resize']), 
        transforms.ToTensor()
    ])
    datasets = {
        k : QualDataset(df[k], preprocessor, transform) for k in ["train", "val"]
    }
    all_comments = ' '.join(datasets['comments']).split()
    vocab = set(all_comments)
    vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    dataloaders = {
        k: DataLoader(datasets[k], batch_size=args['batch_size'], shuffle=True) for k in ["train", "val"]
    }
    
    # GO!
    train(model, dataloaders, args)


if __name__ == "__main__":
    main()