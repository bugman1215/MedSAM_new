import torch
import torch.nn as nn
from PIL import Image
from open_clip import create_model_from_pretrained
import torchvision.transforms as T


# 定义跨模态注意力层
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, img_features, text_features):
        attn_output, attn_weights = self.cross_attention(img_features, text_features, text_features)
        return attn_output, attn_weights


# 修改后的 CLIP 模型，加入 cross-attention
class ModifiedCLIPModel(nn.Module):
    def __init__(self, clip_model, embed_dim, num_heads):
        super(ModifiedCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)
        self.image_projection = nn.Linear(512, embed_dim)
        self.text_projection = nn.Linear(512, embed_dim)

    def forward(self, images, text_input):
        # 获取图像和文本的特征
        image_features = self.clip_model.visual(images)  # 获取图像特征
        text_features = self.clip_model.encode_text(text_input)  # 获取文本特征

        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")

        image_features = image_features.expand_as(text_features)


        # 检查 image_features 和 text_features 的输出维度
        image_features = self.image_projection(image_features)  # 映射到256维

        text_features = self.text_projection(text_features)


        print(f"Projected image features shape: {image_features.shape}")

        # 进行跨模态注意力
        attn_output, attn_weights = self.cross_attention(image_features, text_features)

        return attn_output


# 提取经过跨模态注意力处理后的特征嵌入
def get_clip_embeddings(images, text_inputs) -> torch.Tensor:
    clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    embed_dim = 256
    num_heads = 8

    # 加载 CLIP 模型和预处理函数
    clip_model, clip_preprocess = create_model_from_pretrained(clip_model_name)

    modified_clip_model = ModifiedCLIPModel(clip_model, embed_dim, num_heads)

    # 预处理图像批次
    processed_images = []
    for image in images:  # 假设 images 是一个 batch
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)
        processed_images.append(clip_preprocess(image).unsqueeze(0))  # 处理后的图像添加到列表中

    clip_inputs = torch.cat(processed_images, dim=0)  # 将所有处理后的图像拼接成一个 batch

    # 确保 text_inputs 也是批次的
    if text_inputs.ndim == 2:  # [batch_size, seq_len]
        pass  # 如果是已经处理好的批次，直接使用

    # 使用模型获取跨模态注意力处理后的嵌入
    with torch.no_grad():
        attn_output = modified_clip_model(clip_inputs, text_inputs)  # 处理整个批次
        clip_image_embeddings = attn_output
        clip_prompt_embeddings = clip_image_embeddings.view(clip_image_embeddings.size(0), 1, embed_dim)  # 调整形状

    return clip_prompt_embeddings


