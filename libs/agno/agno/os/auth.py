"""
Authentication Dependencies

Provides FastAPI dependency injection for authentication and authorization.
这是全局的认证依赖，可以在任何需要获取当前用户的地方使用。

使用示例:
    from app.utils.auth.dependencies import get_current_user, require_admin

    @router.get("/protected")
    async def protected_route(current_user: dict = Depends(get_current_user)):
        return {"message": f"Hello {current_user['username']}"}

    @router.get("/admin-only")
    async def admin_route(current_user: dict = Depends(require_admin)):
        return {"message": "Admin access granted"}
"""

from typing import Optional
from fastapi import Depends, Request
from fastapi.security import HTTPBearer

from app.utils.auth.auth_core import AuthCore
from app.utils.auth.auth_response import AuthErrorResponse
from app.utils.auth.auth_errors import AuthErrorType
from app.utils.auth.token_manager import RedisTokenManager
import logging

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(request: Request) -> dict:
    """
    获取当前登录用户（全局依赖注入函数）

    这是核心的认证依赖，使用 AuthCore 进行完整的认证流程：
    1. 提取 token
    2. 验证 token (Redis + JWT)
    3. 加载用户信息
    4. 验证用户状态

    Args:
        request: FastAPI request对象

    Returns:
        用户信息字典，包含：
        - user_id: 用户ID
        - username: 用户名
        - name: 姓名
        - email: 邮箱
        - phone_number: 手机号
        - is_admin: 是否管理员
        - token: 当前token（供刷新使用）
        - _user_model: 完整用户模型

    Raises:
        HTTPException: 认证失败时抛出401/403错误
    """
    token_manager = RedisTokenManager()

    try:
        # 使用 AuthCore 进行完整认证（返回错误类型）
        success, user_context, error_type = await AuthCore.authenticate_request(request, token_manager)

        if not success:
            # 使用错误类型抛出相应的异常
            AuthErrorResponse.from_error_type_exception(error_type)

        return user_context

    except Exception as e:
        # 如果是 HTTPException，直接传递
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            raise

        # 记录意外错误
        logger.error(f"认证过程发生错误: {e}", exc_info=True)
        AuthErrorResponse.from_error_type_exception(AuthErrorType.SERVICE_UNAVAILABLE)


async def require_admin(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    要求管理员权限的依赖

    使用方式：
        @router.get("/admin-only")
        async def admin_route(admin_user: dict = Depends(require_admin)):
            # 这里的admin_user一定是管理员
            pass

    Args:
        current_user: 当前用户信息（来自get_current_user依赖）

    Returns:
        管理员用户信息

    Raises:
        HTTPException: 如果用户不是管理员，返回403 Forbidden
    """
    if not current_user.get("is_admin", False):
        AuthErrorResponse.from_error_type_exception(AuthErrorType.PERMISSION_DENIED)

    return current_user


async def get_optional_user(request: Request) -> Optional[dict]:
    """
    可选的用户认证依赖

    如果提供了有效token则返回用户信息，否则返回None
    适用于可选认证的接口（如：游客可访问，登录用户有额外功能）

    Args:
        request: FastAPI request对象

    Returns:
        用户信息字典或None
    """
    try:
        return await get_current_user(request)
    except Exception:
        # 认证失败时返回 None，不抛出异常
        return None

