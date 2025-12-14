写在前面：
guardagent复现，暂时存储版本（目前能跑，不过只能用gpt，太贵了）
由于deepseek和gpt存在诸多差异（在autogen上）
比如：
autogen期望模型使用function calling API返回代码，但DeepSeek可能在看到这个提示后直接以文本形式输出了代码；
DeepSeek API不支持role为"function"的消息格式（DeepSeek/OpenAI新版API只支持system, user, assistant, tool这几种role）等等

调教了半天Claude都无法解决，遂放弃。跑对一半的代码在：/home/beihang/yx/experiments/self-evoling-agent-experiments/code/code 里
环境就是guardagent。

<center>

# GuardAgent

</center>

## Introduction
This is the repository for the paper [GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning](https://arxiv.org/pdf/2406.09187).

This is an LLM agent designed to act as a guardrail for other LLM agents. Specifically, GuardAgent oversees a target LLM agent by checking whether its inputs/outputs satisfy a set of given guard requests (e.g., safety rules or privacy policies) defined by the users.  

You can find our project page [here](https://guardagent.github.io/).
## Setup
### Packages
We recommend running our code in an environment with `python>=3.9`. Please run
```bash
pip install -r requirements.txt
```
to install necessary packages.
### API Key
Please update your API key in `./config.py`.
### Download Dataset
We support the protection of the EhrAgent based on EICU-AC and the SeeAct based on Mind2Web-SC, and we provide the corresponding dataset. Please refer to [this link](https://github.com/guardagent/dataset) to learn more about and download dataset。
## Run Code
You can run the following command to use our GuardAgent:
```bash
python main.py --llm YOUR_LLM --agent TYOUR_AGENT --seed YOUR_RANDOM_SEED --num_shots YOUR_NUM_SHOTS --logs_path YOUR_LOGS_PATH --dataset_path YOUR_DATASET_PATH
```
Follows are explanations of each parameter:  
`--llm`: The backbone LLM of the GuardAgent you wish to use. You need to set the corresponding API key in `./config.py`.  
`--agent`: The LLM Agent you wish to protect.  Currently, there are two options: `ehragent` and `seeact`.  
`--seed`: The seed used for random shuffle.  
`--num_shots`: The number of shots used by the GuardAgent when generating code. You can choose 1, 2, or 3.  
`--logs_path`: The output path for GuardAgent.  
`--dataset_path`: The path to the dataset that the GuardAgent accepts as input. You should set this parameter to the appropriate path after downloading the dataset (see [Download Dataset](#Download-Dataset)). 

Alternatively, you can set the default values for these options in `./main.py` and then run `python main.py`.
## Citation
If this repository is helpful to you, please consider citing:  
```bibtex
@misc{xiang2024guardagent,
    title={GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning}, 
    author={Zhen Xiang and Linzhi Zheng and Yanjie Li and Junyuan Hong and Qinbin Li and Han Xie and Jiawei Zhang and Zidi Xiong and Chulin Xie and Carl Yang and Dawn Song and Bo Li},
    year={2024},
    eprint={2406.09187},
    archivePrefix={arXiv}}
```
