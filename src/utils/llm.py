"""Helper functions for LLM"""
"""LLM 辅助函数"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and non-Deepseek models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    """
    使用重试逻辑调用 LLM，同时处理 Deepseek 和非 Deepseek 模型。
    
    参数：
        prompt: 发送给 LLM 的提示词
        model_name: 要使用的模型名称
        model_provider: 模型提供者
        pydantic_model: 用于结构化输出的 Pydantic 模型类
        agent_name: 用于进度更新的可选代理名称
        max_retries: 最大重试次数（默认：3）
        default_factory: 失败时创建默认响应的可选工厂函数
        
    返回：
        指定 Pydantic 模型的实例
    """
    from llm.models import get_model, get_model_info
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For non-JSON support models, we can use structured output
    # 对于不支持 JSON 的模型，我们可以使用结构化输出
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    # 调用 LLM 并进行重试
    for attempt in range(max_retries):
        try:
            # Call the LLM
            # 调用 LLM
            result = llm.invoke(prompt)
            
            # For non-JSON support models, we need to extract and parse the JSON manually
            # 对于不支持 JSON 的模型，我们需要手动提取和解析 JSON
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_deepseek_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                # 使用 default_factory 如果提供，否则创建一个基本的默认
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    # 这应该永远不会被上面的重试逻辑所达到
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    """根据模型字段创建安全的默认响应。"""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            # 对于其他类型（如 Literal），尝试使用第一个允许的值
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    # 从 Deepseek 的 Markdown 格式的响应中提取 JSON
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None
