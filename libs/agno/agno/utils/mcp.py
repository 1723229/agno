import json
from functools import partial
from uuid import uuid4

from agno.utils.log import log_debug, log_exception

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
    from mcp.types import Tool as MCPTool
except (ImportError, ModuleNotFoundError):
    raise ImportError("`mcp` not installed. Please install using `pip install mcp`")


from agno.media import Image
from agno.tools.function import ToolResult


# MCP工具名称中文映射表
MCP_TOOL_NAME_ZH = {
    # 知识库相关
    "knowledge_retrieval": "知识库检索",
    "knowledge_search": "知识库搜索",
    "knowledge_query": "知识库查询",

    # 数据分析相关
    "data_agent": "数据分析助手",
    "data_analysis": "数据分析",
    "sql_query": "SQL查询",
    "chart_generation": "图表生成",

    # 文件操作相关
    "file_read": "文件读取",
    "file_write": "文件写入",
    "file_search": "文件搜索",
    "file_upload": "文件上传",

    # 网络相关
    "web_search": "网页搜索",
    "web_scrape": "网页抓取",
    "api_call": "API调用",

    # 代码执行相关
    "python_execute": "Python执行",
    "code_run": "代码运行",
    "shell_command": "Shell命令",

    # 邮件相关
    "email_send": "邮件发送",
    "email_read": "邮件读取",

    # 其他工具
    "calculator": "计算器",
    "translator": "翻译工具",
    "image_generation": "图片生成",
    "text_to_speech": "文字转语音",
}


def get_tool_display_name(tool_name: str) -> str:
    """
    获取工具的中文显示名称

    Args:
        tool_name: 工具的英文名称

    Returns:
        str: 中文名称（如果有映射）或原始英文名称
    """
    return MCP_TOOL_NAME_ZH.get(tool_name, tool_name)


def get_entrypoint_for_tool(tool: MCPTool, session: ClientSession):
    """
    Return an entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use

    Returns:
        Callable: The entrypoint function for the tool
    """

    async def call_tool(tool_name: str, **kwargs) -> ToolResult:
        try:
            await session.send_ping()
        except Exception as e:
            print(e)

        try:
            self_mcp = ['knowledge_retrieval','data_agent']
            if tool_name in self_mcp:
                from app.utils.request_context import RequestContext
                from app.config.settings import ApexConfig

                apex_token = RequestContext.get_access_token()
                file_ids = RequestContext.get_file_ids()
                kb_id = RequestContext.get_kb_id()

                # 如果提供了kb_id，优先使用它
                if file_ids and "faq":
                    # 只有在没有kb_id时才使用FAQ的默认ID
                    file_ids.remove("faq")
                    kwargs.update({"kbIds":[kb_id]})
                    log_debug(f"Using FAQ KB ID from config: {ApexConfig.KB_FAQ_ID}")

                if apex_token:
                    kwargs.update({"apexToken": apex_token})
                    log_debug(f"Using access_token from request context")

                if file_ids:
                    kwargs.update({"folderFileIds": file_ids})
                    log_debug(f"Using file_ids from request context")


            # 获取工具的中文显示名称
            tool_display_name = get_tool_display_name(tool_name)

            log_debug(f"Calling MCP Tool '{tool_display_name}' ({tool_name}) with args: {kwargs}")
            result: CallToolResult = await session.call_tool(tool_name, kwargs)  # type: ignore

            # Return an error if the tool call failed
            if result.isError:
                return ToolResult(content=f"调用工具 '{tool_display_name}' 时出错: {result.content}")

            # Process the result content
            response_str = ""
            images = []

            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    text_content = content_item.text

                    # Parse as JSON to check for custom image format
                    try:
                        parsed_json = json.loads(text_content)
                        if (
                            isinstance(parsed_json, dict)
                            and parsed_json.get("type") == "image"
                            and "data" in parsed_json
                        ):
                            log_debug("Found custom JSON image format in TextContent")

                            # Extract image data
                            image_data = parsed_json.get("data")
                            mime_type = parsed_json.get("mimeType", "image/png")

                            if image_data and isinstance(image_data, str):
                                import base64

                                try:
                                    image_bytes = base64.b64decode(image_data)
                                except Exception as e:
                                    log_debug(f"Failed to decode base64 image data: {e}")
                                    image_bytes = None

                                if image_bytes:
                                    img_artifact = Image(
                                        id=str(uuid4()),
                                        url=None,
                                        content=image_bytes,
                                        mime_type=mime_type,
                                    )
                                    images.append(img_artifact)
                                    response_str += "Image has been generated and added to the response.\n"
                                    continue

                    except (json.JSONDecodeError, TypeError):
                        pass

                    response_str += text_content + "\n"

                elif isinstance(content_item, ImageContent):
                    # Handle standard MCP ImageContent
                    image_data = getattr(content_item, "data", None)

                    if image_data and isinstance(image_data, str):
                        import base64

                        try:
                            image_data = base64.b64decode(image_data)
                        except Exception as e:
                            log_debug(f"Failed to decode base64 image data: {e}")
                            image_data = None

                    img_artifact = Image(
                        id=str(uuid4()),
                        url=getattr(content_item, "url", None),
                        content=image_data,
                        mime_type=getattr(content_item, "mimeType", "image/png"),
                    )
                    images.append(img_artifact)
                    response_str += "Image has been generated and added to the response.\n"
                elif isinstance(content_item, EmbeddedResource):
                    # Handle embedded resources
                    response_str += f"[Embedded resource: {content_item.resource.model_dump_json()}]\n"
                else:
                    # Handle other content types
                    response_str += f"[Unsupported content type: {content_item.type}]\n"

            # Replace <ref> and <image> tags for knowledge_retrieval tool
            if tool_name == 'knowledge_retrieval':
                try:
                    from app.service.base_service import BaseService
                    base_service = BaseService()
                    response_str = await base_service.replace_knowledge_tags(response_str)
                    log_debug("Successfully replaced knowledge tags in response")
                except Exception as e:
                    log_exception(f"Failed to replace knowledge tags: {e}")
                    # Continue with original content if replacement fails

            return ToolResult(
                content=response_str.strip(),
                images=images if images else None,
            )
        except Exception as e:
            tool_display_name = get_tool_display_name(tool_name)
            log_exception(f"Failed to call MCP tool '{tool_display_name}' ({tool_name}): {e}")
            return ToolResult(content=f"调用工具 '{tool_display_name}' 失败: {e}")

    return partial(call_tool, tool_name=get_tool_display_name(tool.name))


def prepare_command(command: str) -> list[str]:
    """Sanitize a command and split it into parts before using it to run a MCP server."""
    import os
    import shutil
    from shlex import split

    # Block dangerous characters
    if any(char in command for char in ["&", "|", ";", "`", "$", "(", ")"]):
        raise ValueError("MCP command can't contain shell metacharacters")

    parts = split(command)
    if not parts:
        raise ValueError("MCP command can't be empty")

    # Only allow specific executables
    ALLOWED_COMMANDS = {
        # Python
        "python",
        "python3",
        "uv",
        "uvx",
        "pipx",
        # Node
        "node",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "bun",
        # Other runtimes
        "deno",
        "java",
        "ruby",
        "docker",
    }

    executable = parts[0].split("/")[-1]

    # Check if it's a relative path starting with ./ or ../
    if executable.startswith("./") or executable.startswith("../"):
        # Allow relative paths to binaries
        return parts

    # Check if it's an absolute path to a binary
    if executable.startswith("/") and os.path.isfile(executable):
        # Allow absolute paths to existing files
        return parts

    # Check if it's a binary in current directory without ./
    if "/" not in executable and os.path.isfile(executable):
        # Allow binaries in current directory
        return parts

    # Check if it's a binary in PATH
    if shutil.which(executable):
        return parts

    if executable not in ALLOWED_COMMANDS:
        raise ValueError(f"MCP command needs to use one of the following executables: {ALLOWED_COMMANDS}")

    first_part = parts[0]
    executable = first_part.split("/")[-1]

    # Allow known commands
    if executable in ALLOWED_COMMANDS:
        return parts

    # Allow relative paths to custom binaries
    if first_part.startswith(("./", "../")):
        return parts

    # Allow absolute paths to existing files
    if first_part.startswith("/") and os.path.isfile(first_part):
        return parts

    # Allow binaries in current directory without ./
    if "/" not in first_part and os.path.isfile(first_part):
        return parts

    # Allow binaries in PATH
    if shutil.which(first_part):
        return parts

    raise ValueError(f"MCP command needs to use one of the following executables: {ALLOWED_COMMANDS}")
