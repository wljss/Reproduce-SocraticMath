import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #防止无法下载的问题


import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

#设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的是cuda/cpu?: {device}")

def load_model_and_tokenizer(): #加载模型和分词器
    model_name = "Qwen/Qwen1.5-4B" #显存不够7B的
    
    #加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='./myModels'
    )
    if tokenizer.pad_token is None: #pad_token是用来填充较短的序列的
        tokenizer.pad_token = tokenizer.eos_token #结束token
    
    #加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto", #自动将模型分布到设备上
        load_in_4bit=True,  #使用4位量化模型压缩技术，减少模型内存占用，稍微降低性能
        trust_remote_code=True, #允许从远程模型代码
        cache_dir='./myModels', #下载的模型储存的位置
    )
    
    return model, tokenizer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_LoRA(model): #初始化LoRA，模型更新W' = W + ΔW中ΔW可以用两个小得多的矩阵B×A表示，B的第二维就是LoRA秩r。冻结原模型，只训练B和A
    
    model = prepare_model_for_kbit_training(model) #准备模型用于k-bit训练
    
    
    lora_config = LoraConfig( #LoRA配置
        r=8, #LoRA秩,很重要的一个参数。r越小训练越快，但拟合能力会下降
        lora_alpha=16, #LoRA alpha参数，用于对ΔW进行缩放,通常为LoRA秩的两倍
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"], #指定在哪些层应用LoRA，分别是注意力机制的Query Projection和Value Projection，前馈神经网络FFN的Gate Projection和Down Projection
        lora_dropout=0.1, #正则化，dropout率，防止过拟合，
        bias="none", #不训练偏置参数
        task_type="QUESTION_ANS", #指定任务类型，这里是问答任务。CAUSAL_LM是对话，后面会试试
    )
    
    model = get_peft_model(model, lora_config) #将LoRA适配器应用到原来的模型上

    print('-----------------下面是可训练的参数-----------------')
    model.print_trainable_parameters() #打印可训练的参数信息
    print('-----------------上面是可训练的参数-----------------')
    
    return model

import pandas as pd

def create_math_teaching_dataset(): #加载数学教学数据集

    
    train_data = pd.read_csv("SocraticMath/data/csv/SocratesMATH.csv", encoding='gbk', encoding_errors ='replace') #好像文件是GB2312编码，但会报错，用gbk也不行，所以encoding_errors ='replace'
    teaching_dialogues= []
    for data in train_data.values:
        tmp = {
            "problem": data[0],
            "dialogue": data[1]
        }
        teaching_dialogues.append(tmp)
    
    return teaching_dialogues

'''
百度的Qwen1.5的对话模板如下:
  <|im_start|>user
  {message}<|im_end|>
  <|im_start|>assistant
  {message}<|im_end|>
'''
def format_data(example): #使用适合对话的格式，格式化数据
    return f"<|im_start|>user\n{example['problem']}<|im_end|>\n<|im_start|>assistant\n{example['dialogue']}<|im_end|>"

from datasets import Dataset

def prepare_dataset(tokenizer): #准备训练数据

    teaching_data = create_math_teaching_dataset() #加载数据
    formatted_texts = [format_data(item) for item in teaching_data] #格式化数据
    
    def tokenize_function(examples): #对文本进行tokenization
        tokenized = tokenizer(
            examples["text"],
            truncation=True, #文本超过max_length时自动截断
            padding=False,
            max_length=1024, #增加长度以容纳对话
            return_tensors=None, #返回Python列表
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy() #对于自回归语言模型，如GPT系列，训练目标是预测序列中的下一个token。因此：输入：[token1, token2, ..., token n-1]，标签：[token2, token3, ..., token n]，但实际上，在Hugging Face的Transformers库中，当labels设置为与input_ids相同时Trainer会自动处理偏移，计算损失时会忽略当前位置对自身的预测。
        return tokenized
    
    dataset = Dataset.from_dict({"text": formatted_texts})#创建数据集
    tokenized_dataset = dataset.map(
        tokenize_function, #应用的函数，对每个批次进行tokenization
        batched=True, #按批次处理数据，而不是逐条处理
        batch_size=128,
        remove_columns=dataset.column_names, #移除原始列，只保留tokenize_function返回的列。最终数据集包含input_ids、attention_mask和labels
    )
    
    return tokenized_dataset

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


if __name__ == "__main__":
    #加载模型和tokenizer
    print("-----------------正在加载模型和分词器-----------------")
    model, tokenizer = load_model_and_tokenizer()
    print("-----------------加载完成-----------------")

    #初始化LoRA
    print("正在初始化LoRA")
    model = setup_LoRA(model)
    print("-----------------初始化完成-----------------")

    #初始化数据集
    print("-----------------正在加载数据集-----------------")
    train_dataset = prepare_dataset(tokenizer)
    print(train_dataset)
    print("-----------------加载完成-----------------")

    #设置训练参数
    print("-----------------正在设置训练参数-----------------")
    training_args = TrainingArguments(
        output_dir="./qwen1.5-4b-math-teacher", #输出目录
        per_device_train_batch_size=1, #每个设备每次前向传播处理的批量大小
        gradient_accumulation_steps=8, #累积多少次的梯度然后更新权重
        num_train_epochs=5, #训练轮数
        learning_rate=1e-4, #学习率
        fp16=True, #启用混合精度训练，有的时候用16位浮点，有的时候32位，减少显存使用
        logging_steps=10, #训练日志记录间隔
        save_steps=200, #每训练200步保存一次检查点
        eval_steps=200, #每200步在验证集上评估一次
        save_total_limit=3, #最多只保留3个最新的检查点
        remove_unused_columns=False, #保留数据集中所有列
        run_name="qwen1.5-4b-math-teacher",
        report_to=None, #不向任何平台报告训练进度
        warmup_steps=100, #预热步数，在训练开始时线性增加学习率，防止训练初期梯度爆炸
        lr_scheduler_type="cosine", #使用余弦退火学习率调度
    )
    print("-----------------设置完成-----------------")
    
    #创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, #mlm=True用于BERT等模型的训练，随机掩盖部分token让模型预测，我们这里需要关闭
    )
    
    #创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    #开始训练
    print("-----------------开始训练-----------------")
    trainer.train()
    
    #保存模型
    print("-----------------训练结束，正在保存模型-----------------")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("-----------------训练完成！-----------------")