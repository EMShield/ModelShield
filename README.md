# EMShield

EMShield provides a full-lifecycle security framework for the secure on-device deployment of small language models (SLMs) across edge device clusters.

# Features Overview

1. **Trusted On-Device Deployment of SLMs**, achieved through the following components:

  - **LoRA Parameter Decoupling**: Decouples core model weights to enable differentiated protection strategies for critical parameters.
  - **Edge-Cloud Trusted Collaboration**: Minimizes the exposure surface of sensitive weights during deployment. Each set of decoupled weights is traceably bound to a specific edge user, enabling publicly verifiable provenance.
  - **Defense and Protection**: Incorporates robust fingerprints and watermarks to ensure the integrity and continuity of the traceability chain under adversarial conditions.

2. **Hardware-Software Co-Design for Security Assurance**:

  - Core weights are stored within a **Trusted Execution Environment (TEE)**, preventing unauthorized access. Dynamic mounting of the TEE further strengthens this isolation.
  - The **LoRA-based parameter decoupling** transforms the inversion of full model weights into a computationally intractable problem—namely, the unique decomposition of low-rank matrices, which is NP-hard.
  - A **chaos-map-driven finite state machine (FSM)** supplies high-entropy randomness for fingerprint and watermark embedding. Additionally, defense mechanisms tied to device-specific UUIDs render these marks publicly verifiable.

![image-20250515125621629](https://typora-hui.oss-cn-chengdu.aliyuncs.com/img/20250515125626891.png)


# Quick Start

## LoRA Parameter Decoupling

![image-20250515135203999](https://typora-hui.oss-cn-chengdu.aliyuncs.com/img/20250515135204058.png)

​	This module partitions the SLM weights into general-purpose inference backbone weights and downstream-task-specific core weights, thereby achieving secure isolation of task-critical parameters and minimizing unnecessary exposure. The overall workflow comprises four stages:

* **Initialize**：Prepares the execution environment for subsequent weight decoupling.
* **Data processing**：Reformats the training dataset into a structure suitable for decoupling.
* **Training**：Performs LoRA-based fine-tuning on the structured dataset to derive the decoupled model weights.
* **Integrate Weight**：Reconstructs the original weight composition to enable inference for specific downstream tasks.

### Initialize

The weight decoupling process is implemented based on [LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/index.html). The following commands can be executed sequentially to install the framework locally:

1. **Install LLaMA-Factory and Dependencies**: 
   To install LLaMA-Factory and its dependencies, run the following command. If dependency conflicts occur, use the `--no-deps` flag to bypass them:

   ```bash
   cd M1-LoRA Parameter Decoupling
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   ```

2. **Verify Installation**:

   Ensure that the installation completes successfully by executing a test run or importing the core modules without errors.

   ```bash
   llamafactory-cli version
   ```



### Data processing

We construct four instruction-supervised fine-tuning datasets in the required format:

- **financial.json**: A financial lending dataset generated using the Faker library, containing 10,000 synthetic loan records. Each entry includes a user’s profile and the corresponding loan approval decision.
- **identifier.json**: A privacy identification dataset designed to extract quasi-identifiers and direct identifiers from user data. It includes 10,000 records, each containing multiple types of personal identifiers, generated using the Faker library.
- **law.json**: A legal judgment dataset based on the DISC-Law-SFT dataset released in 2023 by Fudan University's Data Intelligence and Social Computing Lab (FudanDISC).
- **medical.json**: A medical QA dataset sourced from Huatuo-26M [53], the largest Chinese-language medical question-answering corpus, comprising 26 million QA pairs.

For example:

```json
{
  "instruction": "Now you are an otolaryngologist. Please provide advice based on the patient's question:",
  "input": "I had a cold last month and didn't take any medication. After the cold recovered, I felt my nose was often blocked, with frequent redness, itching of the nasal skin, and dizziness. I always thought it was a sequela from the previous cold. After going to the hospital for examination, the result showed nasal septum deviation. How to treat nasal septum deviation?",
  "output": "There are two methods for treating nasal septum deviation: surgical and non-surgical treatments. Surgical treatment corrects the nasal septum deviation through surgery, while non-surgical treatment relieves symptoms through medication and physical therapy. Surgery is the most effective method for treating nasal septum deviation. After surgery, it is necessary to rest, avoid strenuous exercise and head-down work, and pay attention to diet by eating less spicy food and avoiding alcohol. It is normal to have blood or blood clots in nasal discharge or sputum within two weeks after surgery. Seek medical attention immediately if there is heavy bleeding, fever, or severe pain."
}
```



### Training

First, launch the **LLaMA-Factory WebUI** to execute the training process.

```bash
llamafactory-cli webui
```

Import the prepared training dataset and manually configure the following training hyperparameters:

- **Model Name and Path**, **Training Stage**, **Fine-Tuning Method**, **Training Dataset**
- **Learning Rate**, **Number of Training Epochs**, and other optimization parameters
- **LoRA-specific Parameters** and additional fine-tuning configurations
- **Output Directory** and **Configuration File Path**

Upon completion of the training process, click *Export* to obtain the decoupled core weights. For instance, `finance_A.pt` and `finance_B.pt` correspond to the decomposed A and B matrices of the original weight file `financial.pt`.



### Integrate Weight

Execute the following command to merge the model weights:

```bash
llamafactory-cli export merge_config.yaml
```

Here, `merge_config.yaml` specifies the parameters required for the merging process. A sample configuration file is shown below:

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3.2-1B-Instruct
adapter_name_or_path: saves/llama3.2-1b/lora/sft
template: llama3.2
finetuning_type: lora

### export
export_dir: models/llama3.2_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

## Edge-Cloud Trusted Collaboration

![image-20250515140555468](https://typora-hui.oss-cn-chengdu.aliyuncs.com/img/20250515140555575.png)

​	This module stores the SLM’s core weights within a **Trusted Execution Environment (TEE)** while deploying the backbone model on the open-source [community platform](https://modelscope.cn/home). Controlled model deployment services are provided to edge-side users. The overall process is divided into two stages:

- **Trusted Storage**: Minimizes the attack surface of core weights through dynamic nested mounting and unmounting.
- **Model Deploy**: Supports fine-grained deployment partitioning and audits critical steps during deployment.

### Initialize the TEE environment

Run the following command to create the TEE environment for storing core weights, using an edge-side service partition as an example:

```bash
cd M2-Edge-Cloud Trusted Collaboration/PrototypeSystem/edgeSide_business

dd if=/dev/zero of=modelSide_area_vfs bs=1G count=3

sudo losetup /dev/loop1 ./modelSide_area_vfs

sudo cryptsetup luksFormat /dev/loop1

sudo cryptsetuo luksOpen /dev/loop1 modelside_area_mapper

sudo mkfs.ext4 /dev/mapper/modelside_area_mapper
```

Additionally, model service partitioning involves complex nested mounting. The mounting script is shown below:

```bash
#!/bin/bash
key_id=$(keyctl search @s user my_persistent_key)
key_value=$(keyctl pipe $key_id)

echo $key_value | sudo -S cryptsetup luksOpen /dev/loop1 modelside_area_mapper
sudo mount /dev/mapper/modelside_area_mapper /home/cloud_server/modelSide_business

sudo losetup /dev/loop2 /home/cloud_server/modelSide_business/precision_core_vfs
sudo losetup /dev/loop3 /home/cloud_server/modelSide_business/not_precision_core_vfs
echo $key_value | sudo -S cryptsetup luksOpen /dev/loop2 precision_core_mapper
echo $key_value | sudo -S cryptsetup luksOpen /dev/loop3 not_precision_core_mapper
sudo mount /dev/mapper/precision_core_mapper /home/cloud_server/modelSide_business/precision_core_mount
sudo mount /dev/mapper/not_precision_core_mapper /home/cloud_server/modelSide_business/not_precision_core_mount
```



### Start the frontend and backend services

First, expose the model deployment service using **FRP for intranet penetration**:

```bash
## ===Start backend===
cd M2-Edge-Cloud Trusted Collaboration/PrototypeSystem/edgeSide_business/backend
sudo -u nodejs pm2 start server.js --name backend_jy
pm2 start frps --name frps_jy -- -c M2-Edge-Cloud Trusted Collaboration/PrototypeSystem/edgeSide_business/frps.ini
pm2 start frpc --name frpc_jy -- -c M2-Edge-Cloud Trusted Collaboration/PrototypeSystem/edgeSide_business/frpc.ini
sudo -u nodejs pm2 flush backend
sudo -u nodejs pm2 logs backend
## ===Start frontend===
cd M2-Edge-Cloud Trusted Collaboration/PrototypeSystem/edgeSide_business/frontend
npm run build
systemctl restart nginx
```

Key configuration elements include:

- **Frontend service port configuration**: `server.js`

- **FRP tunneling settings**: `frps.ini`, `frpc.ini`

- **Redirect backbone deployment route**: `/services/main_Model.js`
   → Target backbone deployment: [Backbone Model](https://modelscope.cn/models/Huiyuchen/hui_model)

- **Core weight deployment setup**: `/services/main_Model.js`
   → `downloadCoreFiles()`

  - ```js
    async downloadCoreFiles(uuid, modelIndex) {
            try {
                ....
                await this.mountScriptPromise;
                const mountPoint = '/home/cloud_server/nxg/modelSide_business';
                const ptFile = path.join(mountPoint, 'precision_area_mount', `model_${index}.pt`);
                const mdFile = path.join(mountPoint, `model_${index}.md`);
                await Promise.all([
                    fs.access(ptFile, fs.constants.R_OK),
                    fs.access(mdFile, fs.constants.R_OK)
                ]);
                return {
                    success: true,
                    message: '核心权重文件已准备就绪，开始下载',
                    modelIndex: index,
                    files: {
                        // coreWeight
                    }
                };
            } catch (error) {
                ....
            }
        }
    ```

    

## Defense and Protection

![image-20250515145317702](https://typora-hui.oss-cn-chengdu.aliyuncs.com/img/20250515145317765.png)

This module determines whether to apply **model watermarking** or **model fingerprinting** for provenance tracing based on the type of core weights. The process is divided into three main stages:

- **Matrix Decomposition Model Training**: Trains a model to reconstruct watermark information from the integrated matrix back into the decoupled core weights produced by the previous module.
- **Model Watermark Embedding**: Embeds a 48-bit watermark into the core weights of the model, suitable for models with low sensitivity to inference accuracy.
- **Model Fingerprint Extraction**: Extracts invariant characteristics from the core weights as fingerprints, suitable for models with high sensitivity to inference accuracy.	

### Matrix Factorization Model with Prior Knowledge Training

Run the following command to start training:

```bash
cd M3-Defense and Protection/MatrixSplitModel
python train_model.py
```

Checkpoints generated during training are saved in: M3-Defense and Protection/MatrixSplitModel/models.

The **pretrained weights** for the matrix decomposition model can be obtained [here](https://drive.google.com/file/d/1NNls_2Wr0QXoif9pSL0CZGiPw2X0wHHF/view?usp=sharing).



### Watermark Embedding & Fingerprint Extraction

Run the following command to apply model watermarking and fingerprinting:

```bash
cd M3-Defense and Protection/protectionPolicy
 
 # in_watermark.ipynb: 读取lora_weights.pt，并将指定的水印信息嵌入到映射之后的核心权重中，并输出嵌入水印之后的模型权重
 # in_fingerPrint.ipynb：读取lora_weights.pt，读取模型核心权重中的不变项
```

***Note:*** Please download the [pretrained weights](https://drive.google.com/file/d/1b4hVZCU5Zh7q1WyDqedYiO3ug-OBQF7q/view?usp=sharing) and place them under the following directory:/protectionPolicy/models/

# Contributing

Anyone is welcome to provide any form of contribution, for example:

- More weight decoupling methods for SLM (separation of core weights and ordinary weights)
- More robust workflows for watermarking and fingerprinting
- More usage scenarios
- Documentation, bug fixes, security improvements
- Others ...

Please check [CONTRIBUTING.md](CONTRIBUTING.md).


# License

Please check [LICENSE](LICENSE) for details.