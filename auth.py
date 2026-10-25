import os
import sqlite3
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from config import settings

logger = logging.getLogger("rag_agent.auth")

DB_PATH = settings.BASE_DIR / "users.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Initializes the SQLite users database and creates the users table if not exists."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _hash_password(password: str, salt: str) -> str:
    """Hashes password with salt using PBKDF2 HMAC SHA-256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100000,
    ).hex()


def register_user(username: str, password: str, display_name: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Registers a new user account.
    Returns: (success: bool, user_data: Optional[dict], message: str)
    """
    init_auth_db()
    uname = (username or "").strip().lower()
    if not uname or len(uname) < 3:
        return False, None, "Username must be at least 3 characters long."
    if not password or len(password) < 6:
        return False, None, "Password must be at least 6 characters long."

    name = (display_name or "").strip() or uname.capitalize()
    user_id = secrets.token_hex(12)
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)

    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users (id, username, display_name, password_hash, salt) VALUES (?, ?, ?, ?, ?)",
                (user_id, uname, name, pwd_hash, salt),
            )
            conn.commit()

        user_info = {
            "id": user_id,
            "username": uname,
            "display_name": name,
        }
        logger.info(f"Registered new user: '{uname}' (id: {user_id})")
        return True, user_info, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, None, f"Username '{uname}' is already taken."
    except Exception as e:
        logger.error(f"User registration error: {e}")
        return False, None, f"Registration failed: {str(e)}"


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates an existing user with username and password.
    Returns: (success: bool, user_data: Optional[dict], message: str)
    """
    init_auth_db()
    uname = (username or "").strip().lower()
    if not uname or not password:
        return False, None, "Please enter both username and password."

    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, username, display_name, password_hash, salt FROM users WHERE username = ?",
                (uname,),
            )
            row = cursor.fetchone()

        if not row:
            return False, None, "Invalid username or password."

        stored_hash = row["password_hash"]
        salt = row["salt"]
        calc_hash = _hash_password(password, salt)

        if secrets.compare_digest(stored_hash, calc_hash):
            user_info = {
                "id": row["id"],
                "username": row["username"],
                "display_name": row["display_name"],
            }
            logger.info(f"Authenticated user: '{uname}'")
            return True, user_info, "Login successful!"
        else:
            return False, None, "Invalid username or password."
    except Exception as e:
        logger.error(f"User authentication error: {e}")
        return False, None, f"Authentication error: {str(e)}"


def get_user_count() -> int:
    """Returns the total number of registered users."""
    init_auth_db()
    try:
        with _get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]
    except Exception:
        return 0


def render_auth_ui():
    """
    Renders the modern sign-in / registration screen in Streamlit.
    """
    import streamlit as st

    st.markdown(
        """
        <div style="max-width: 520px; margin: 2rem auto 1.5rem auto; padding: 2rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 16px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4); text-align: center;">
            <div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ RAG AI Agent Studio</div>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0;">Multi-Tenant Secure Document Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab_login, tab_signup = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            with st.form("form_login"):
                uname = st.text_input("Username", key="login_uname", placeholder="e.g. praveen")
                pwd = st.text_input("Password", type="password", key="login_pwd")
                btn_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if btn_login:
                    success, user, msg = authenticate_user(uname, pwd)
                    if success and user:
                        st.session_state["user"] = user
                        st.toast(f"Welcome back, {user['display_name']}! 👋")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_signup:
            with st.form("form_register"):
                new_uname = st.text_input("Choose Username", key="reg_uname", placeholder="e.g. praveen")
                new_name = st.text_input("Display Name", key="reg_name", placeholder="e.g. Praveen Kumar")
                new_pwd = st.text_input("Choose Password (min 6 chars)", type="password", key="reg_pwd")
                btn_reg = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if btn_reg:
                    success, user, msg = register_user(new_uname, new_pwd, new_name)
                    if success and user:
                        st.session_state["user"] = user
                        st.toast(f"Account created! Welcome, {user['display_name']}! 🎉")
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("<div style='text-align: center; margin-top: 1.25rem;'>", unsafe_allow_html=True)
        if st.button("🚀 Continue as Guest", key="btn_guest", use_container_width=True):
            st.session_state["user"] = {
                "id": "guest",
                "username": "guest",
                "display_name": "Guest Explorer",
            }
            st.rerun()
        st.caption("🔒 Guest mode provides shared access. Sign in to isolate your uploaded documents.")
        st.markdown("</div>", unsafe_allow_html=True)
