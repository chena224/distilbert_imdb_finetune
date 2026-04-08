from transformers import AutoModelForMaskedLM
model_checkpoint="distilbert-base-uncased"
model=AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.num_parameters()

# 稍微使用一下模型，英语维基百科做的预训练
text="This is a great [MASK]."

from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

inputs=tokenizer(text,return_tensors="pt")
logits=model(**inputs).logits
mask_token_index=torch.where(inputs["input_ids"]==tokenizer.mask_token_id)[1]
mask_token_logits=logits[0,mask_token_index,:]
top_5_tokens=torch.topk(mask_token_logits,5,dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(f"{text.replace(tokenizer.mask_token,tokenizer.decode([token]))}")

# 利用大型电影数据集IMDb开始微调

from datasets import load_dataset
imdb_dataset=load_dataset('imdb')

# 数据情况： DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     unsupervised: Dataset({
#         features: ['text', 'label'],
#         num_rows: 50000
#     })
# })


# 定义数据处理,后续会做填充截断处理，这里不输入参数，也不使用datacollator
def tokenize_function(example):
    result=tokenizer(example["text"])
    if tokenizer.is_fast:
        result["word_ids"]=[result.word_ids[i] for i in range(len(result["input_ids"]))]
    return result

# map流水线，防止内存爆炸
tokenized_datasets=imdb.map(tokenize_fuction,batched=True,remove_columns=["text","label"])

# 处理后情况：DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'input_ids', 'word_ids'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['attention_mask', 'input_ids', 'word_ids'],
#         num_rows: 25000
#     })
#     unsupervised: Dataset({
#         features: ['attention_mask', 'input_ids', 'word_ids'],
#         num_rows: 50000
#     })
# })


# 原模型上下文窗口长度512
tokenizer.model_max_length 

# 自定义截断，可设置512，根据gpu情况选择
chunk_size=128
def group_texts(examples):
    concatenated_examples={k:sum([examples[k],[]]) for k in examples.keys()}
    total_length=len(concatenated_examples(list[(examples.keys())[0]]))
    # 剩余部分舍弃
    total_length=(total_length//chunk_size)*chunk_size
    result={k:[t[i:i+chunk_size] for i in range(0,total_length,chunk_size)]
    for k,t in concatenated_examples.items()
    }
# 微调时输出最好不单以mask的输出或者其他词-100，应全词输出，但后续使用像预训练仅以mask为输出，其他为-100
    result["labels"]=result["input_ids"].copy()
    return result

# map批量处理，防止数据庞大导致内存爆炸
lm_datasets=tokenized_datasets.map(group_texts,batched=True)

# lm_datasets情况：
# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 61289
#     })
#     test: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 59905
#     })
#     unsupervised: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 122963
#     })
# })

# 1.token直接填充，后续会写自定义的全词填充
from transformers import DataCollatorForlanguageLM
data_collator=DataCollatorForLanguageLM(tokenizer=tokenizer,mlm_probability=0.15)

# 2全词填充
import collections
import numpy as np
from transformers import default_data_collator

wwm_probability=0.2

# 此处采取预训练逻辑，label只mask实际对应id输出，其他-100，忽略。
def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids=features.pop("word_ids")

        mapping=collections.defaultdict(list)
        current_word_index=-1
        current_word=None
        for idx,word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id!=current_word:
                   current_word=word_id
                   current_word_index+=1
                mapping[current_word_index].append(idx)
        mask=np.random.binomial(1,wwm_probability,len(mapping,))
        input_ids[idx]=feature["input_ids"]
        labels=feature["labels"]
        new_labels=[-100]*len(labels)
        for word_idx in np.where(mask)[0]:
            word_idx=word_idx.item()
            for idx in mapping[word_idx]:
                new_labels[idx]=labels[idx]
                input_idx=tokenizer_mask_token_id
        feature["labels"]=new_labels
    return default_data_collator(features)

# 设置训练集与验证集
train_size=10000
test_size=int(0.1*train_size)
downsampled_dataset=lm_datasets["train"].train_test_split(
    train_size=train_size,test_size=test_size,seed=42
)
# downsampled_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 10000
#     })
#     test: Dataset({
#         features: ['attention_mask', 'input_ids', 'labels', 'word_ids'],
#         num_rows: 1000
#     })
# })

# 这里进行了huggingface的登陆并上传hub操作，因本人学习阶段，无huggingface_hub上传操作,d但仍会原样打出
from huggingface_hub import notebook_login
notebook_login()

# 此处用trainer，后面因为评估结果可复现会修改、
from transformers import TrainingArguments
batch_size=64
logging_steps=len(downsampled_dataset['train'])//batch_size
model_name=model_checkpoint.split("/")[-1]

# 这里使用混合精度，loss时候16，存梯度时32，合理降低精度，降低内存，加速度
training_args=TrainingArguments(
    output_dir=f'{model_name}-finetuned_imdb',
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # 此处仅因学习阶段，故完全复现代码
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps
)

from transformers import Trainer
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
eval_results=trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.push_to_hub()

# 使用accelerator,同时保证评估结果困惑度可复现

def insert_random_mask(batch):
    features=[dict(zip(batch,t)) for t in zip(*batch.values())]
    masked_inputs=data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

# loader分批次加载batch
from torch.utils.data import DataLoader
from transformers import default_data_collator
batch_size=64
train_dataloader=Dataloader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
# 此处collator_fn填default_data_collator不在让其掩码
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

from accelerate import Accelerator

# 一站式弄好配置
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# user/huggingface/仓库名/model_name
from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name

# 本地与云端链接，
from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

# 训练与评估循环
from tqdm.auto import tqdm
import torch
import math

progress_bar=tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs=model(**batch)
        loss=outputs.loss
        # 混合精度
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    model.eval()
    losses=[]
    for step,batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs=model(**batch)
        
        loss=outputs.loss
        # 分布式，所以收集所有gpu的loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))
    losses=torch.cat(losses)
    losses=losses[:len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

#    gpu全部结束
    accelerator.wait_for_everyone()
    # 保存去掉accelerator的原始模型权重
    unwrapped_model = accelerator.unwrap_model(model)
    # 主块写入
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


# 上传成功后直接调用
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
# '>>> this is a great movie.'
# '>>> this is a great film.'
# '>>> this is a great story.'
# '>>> this is a great movies.'
# '>>> this is a great character.'