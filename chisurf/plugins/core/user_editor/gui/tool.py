import os
import uuid
import yaml

from chisurf import logging
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QStatusBar, QHeaderView, QGroupBox, QFormLayout,
    QComboBox, QCheckBox, QDialog, QProgressBar, QSizePolicy
)
from qtpy.QtCore import Qt

import chisurf.core.settings as cs_settings


class PasswordChangeDialog(QDialog):
    """
    A dialog for setting/changing a password with strength indicator.
    Calculates and shows password strength using color coding (red, yellow, green).
    """
    def __init__(self, user_id: str, is_admin: bool, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.is_admin = is_admin
        self.setWindowTitle(f"Change Password for {user_id}")
        self.resize(360, 240)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        form_layout = QFormLayout()
        self.edit_password = QLineEdit()
        self.edit_password.setEchoMode(QLineEdit.Password)
        self.edit_password.textChanged.connect(self.on_password_changed)
        form_layout.addRow("New Password:", self.edit_password)
        
        self.edit_confirm = QLineEdit()
        self.edit_confirm.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Confirm Password:", self.edit_confirm)
        
        layout.addLayout(form_layout)
        
        # Strength indicator
        self.strength_label = QLabel("Strength: Weak")
        self.strength_label.setStyleSheet("color: #ff4d4d; font-weight: bold;")
        layout.addWidget(self.strength_label)
        
        self.strength_bar = QProgressBar()
        self.strength_bar.setRange(0, 100)
        self.strength_bar.setValue(0)
        self.strength_bar.setTextVisible(False)
        self.strength_bar.setMaximumHeight(8)
        self.strength_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4d4d; }")
        layout.addWidget(self.strength_bar)
        
        # Feedback label
        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("color: #555555; font-size: 11px;")
        layout.addWidget(self.feedback_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.on_password_changed("")
        
    def on_password_changed(self, text):
        score = 0
        feedback = []
        
        if len(text) >= 8:
            score += 1
        else:
            feedback.append("At least 8 characters")
            
        if any(c.islower() for c in text):
            score += 1
        else:
            feedback.append("At least one lowercase letter")
            
        if any(c.isupper() for c in text):
            score += 1
        else:
            feedback.append("At least one uppercase letter")
            
        if any(c.isdigit() for c in text):
            score += 1
        else:
            feedback.append("At least one number")
            
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        if any(c in special_chars for c in text):
            score += 1
        else:
            feedback.append("At least one special character")
            
        self.score = score
        self.strength_bar.setValue(score * 20)
        
        if not text:
            self.strength_bar.setValue(0)
            self.strength_bar.setStyleSheet("QProgressBar::chunk { background-color: #cccccc; }")
            self.strength_label.setText("Enter a password")
            self.strength_label.setStyleSheet("color: #777777;")
            self.feedback_label.clear()
        elif score <= 2:
            self.strength_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4d4d; }")
            self.strength_label.setText("Strength: Weak")
            self.strength_label.setStyleSheet("color: #ff4d4d; font-weight: bold;")
            self.feedback_label.setText("Requirements: " + ", ".join(feedback))
        elif score <= 4:
            self.strength_bar.setStyleSheet("QProgressBar::chunk { background-color: #ffc107; }")
            self.strength_label.setText("Strength: Medium")
            self.strength_label.setStyleSheet("color: #ffc107; font-weight: bold;")
            self.feedback_label.setText("Requirements: " + ", ".join(feedback))
        else:
            self.strength_bar.setStyleSheet("QProgressBar::chunk { background-color: #28a745; }")
            self.strength_label.setText("Strength: Strong")
            self.strength_label.setStyleSheet("color: #28a745; font-weight: bold;")
            self.feedback_label.clear()
            
    def on_save_clicked(self):
        password = self.edit_password.text()
        confirm = self.edit_confirm.text()
        
        if password != confirm:
            QMessageBox.warning(self, "Validation Error", "Passwords do not match.")
            return
            
        if self.is_admin and self.score < 4:
            QMessageBox.warning(self, "Validation Error", "Administrator password is too weak. It must be at least Medium strength (score >= 4).")
            return
            
        self.password = password
        self.accept()


class UserEditorWidget(QWidget):
    """
    A GUI plugin for managing users in Chisurf coupled with MFDB.
    Communicates with the backend using the ZMQ JSON-RPC client,
    ensuring conformance to the new plugin architecture.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("User Editor")
        self.resize(980, 520)
        self.setMinimumSize(760, 420)

        self.users = []
        self.selected_user_id = None
        self.is_creating_new = False

        self.setup_ui()
        self.load_users()

    def setup_ui(self):
        # Outer layout to support QHBoxLayout + Status Bar in a QWidget
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(4)

        # Horizontal layout for left/right panels
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)
        outer_layout.addLayout(main_layout, stretch=1)

        # Left panel: user list
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        
        list_title = QLabel("Available Users")
        font = list_title.font()
        font.setBold(True)
        list_title.setFont(font)
        left_layout.addWidget(list_title)

        # Users table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Active", "Display Name", "User ID"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setMinimumHeight(160)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.table.itemSelectionChanged.connect(self.on_user_selection_changed)
        left_layout.addWidget(self.table, stretch=1)

        # Left panel buttons
        left_buttons_layout = QHBoxLayout()
        self.btn_new = QPushButton("New User")
        self.btn_new.clicked.connect(self.on_new_user_clicked)
        self.btn_delete = QPushButton("Delete User")
        self.btn_delete.clicked.connect(self.on_delete_user_clicked)
        left_buttons_layout.addWidget(self.btn_new)
        left_buttons_layout.addWidget(self.btn_delete)
        left_layout.addLayout(left_buttons_layout)

        main_layout.addLayout(left_layout, stretch=3)

        # Right panel: user details and actions
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        details_group = QGroupBox("User Details")
        form_layout = QFormLayout(details_group)
        form_layout.setContentsMargins(6, 6, 6, 6)
        form_layout.setSpacing(6)

        self.edit_user_uuid = QLineEdit()
        self.edit_user_uuid.setReadOnly(True)
        self.edit_user_uuid.setStyleSheet("background-color: #e6e6e6; color: #555555;")
        form_layout.addRow("User UUID:", self.edit_user_uuid)

        self.edit_user_id = QLineEdit()
        form_layout.addRow("Username *:", self.edit_user_id)

        self.edit_display_name = QLineEdit()
        form_layout.addRow("Display Name *:", self.edit_display_name)

        self.edit_email = QLineEdit()
        form_layout.addRow("Email:", self.edit_email)

        self.edit_role = QComboBox()
        self.edit_role.addItems([
            "Generic",
            "Principal Investigator",
            "Postdoc",
            "PhD Student",
            "Master Student",
            "Bachelor Student",
            "Technician",
            "Industry Professional",
            "Scientist",
            "Manager",
            "Other"
        ])
        form_layout.addRow("Role:", self.edit_role)

        self.edit_affiliation = QLineEdit()
        form_layout.addRow("Affiliation:", self.edit_affiliation)

        self.edit_department = QLineEdit()
        form_layout.addRow("Department:", self.edit_department)

        self.edit_phone = QLineEdit()
        form_layout.addRow("Phone:", self.edit_phone)

        self.edit_website = QLineEdit()
        form_layout.addRow("Website:", self.edit_website)

        self.edit_is_admin = QCheckBox("Is Administrator")
        self.edit_is_admin.toggled.connect(self._apply_admin_autologin_rule)
        form_layout.addRow("", self.edit_is_admin)

        self.edit_allow_autologin = QCheckBox("Allow autologin")
        self.edit_allow_autologin.setToolTip(
            "Allow this user to obtain a login session without entering a password."
        )
        form_layout.addRow("", self.edit_allow_autologin)

        self.btn_change_password = QPushButton("Change Password...")
        self.btn_change_password.clicked.connect(self.on_change_password_clicked)
        form_layout.addRow("Password:", self.btn_change_password)

        self.edit_address = QTextEdit()
        self.edit_address.setMaximumHeight(50)
        form_layout.addRow("Address:", self.edit_address)

        self.edit_details = QTextEdit()
        self.edit_details.setMaximumHeight(50)
        form_layout.addRow("Details:", self.edit_details)

        right_layout.addWidget(details_group)

        # Right panel actions
        right_buttons_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Changes")
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.btn_set_active = QPushButton("Set as Active User")
        self.btn_set_active.clicked.connect(self.on_set_active_clicked)
        right_buttons_layout.addWidget(self.btn_save)
        right_buttons_layout.addWidget(self.btn_set_active)
        right_layout.addLayout(right_buttons_layout)
        right_layout.addStretch()

        main_layout.addLayout(right_layout, stretch=2)

        # Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        outer_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")

    def load_users(self, select_user_id: str | None = None):
        """Fetch users from the database and populate the table.

        Parameters
        ----------
        select_user_id : str, optional
            Username to reselect after reloading.
        """
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(0)

        try:
            client = self.make_mfdb_client()
            logging.info("User Editor: loading users via MFDB RPC")
            self.users = client.list_users()
            logging.info("User Editor: loaded %d users", len(self.users))
        except Exception as e:
            logging.exception("User Editor: failed to load users via MFDB RPC")
            QMessageBox.critical(self, "ZMQ RPC Error", f"Could not load users via JSON-RPC:\n{e}")
            self.users = []

        active_id = cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")

        self.table.setRowCount(len(self.users))
        for i, user in enumerate(self.users):
            is_active = user["user_id"] == active_id
            active_text = "★" if is_active else ""
            
            item_active = QTableWidgetItem(active_text)
            item_active.setTextAlignment(Qt.AlignCenter)
            item_active.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            item_name = QTableWidgetItem(user["display_name"])
            item_name.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            item_id = QTableWidgetItem(user["user_id"])
            item_id.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.table.setItem(i, 0, item_active)
            self.table.setItem(i, 1, item_name)
            self.table.setItem(i, 2, item_id)

        self.is_creating_new = False
        selected_row = None
        if select_user_id:
            for row, user in enumerate(self.users):
                if user["user_id"] == select_user_id:
                    selected_row = row
                    break
        if selected_row is None:
            self.table.blockSignals(False)
            self.clear_details()
        else:
            self.table.selectRow(selected_row)
            self.table.blockSignals(False)
            self.on_user_selection_changed()
        self.status_bar.showMessage(f"Loaded {len(self.users)} users. Active user: {active_id}")

    def clear_details(self):
        self.edit_user_uuid.clear()
        self.edit_user_id.clear()
        self.edit_user_id.setReadOnly(True)
        self.edit_user_id.setStyleSheet("background-color: #e6e6e6; color: #555555;")
        self.edit_display_name.clear()
        self.edit_email.clear()
        self.edit_role.setCurrentIndex(0)
        self.edit_affiliation.clear()
        self.edit_department.clear()
        self.edit_phone.clear()
        self.edit_website.clear()
        self.edit_is_admin.setChecked(False)
        self.edit_is_admin.setEnabled(True)
        self.edit_allow_autologin.setChecked(False)
        self.edit_allow_autologin.setEnabled(True)
        self.btn_change_password.setEnabled(True)
        self.temp_password = None
        self.edit_address.clear()
        self.edit_details.clear()
        self.selected_user_id = None
        self.btn_set_active.setEnabled(False)

    def on_user_selection_changed(self):
        """Populate right panel fields when user is selected in table."""
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            return

        row = selected_ranges[0].topRow()
        if row < 0 or row >= len(self.users):
            return

        self.is_creating_new = False
        user = self.users[row]
        self.selected_user_id = user["user_id"]
        
        self.edit_user_uuid.setText(user.get("user_uuid") or "")
        self.edit_user_id.setText(user["user_id"])
        can_rename = user["user_id"] not in ("user_default", "guest")
        self.edit_user_id.setReadOnly(not can_rename)
        self.edit_user_id.setStyleSheet("" if can_rename else "background-color: #e6e6e6; color: #555555;")
        self.edit_display_name.setText(user["display_name"] or "")
        self.edit_email.setText(user["email"] or "")
        
        role_val = user.get("role") or "Generic"
        idx = self.edit_role.findText(role_val)
        if idx >= 0:
            self.edit_role.setCurrentIndex(idx)
        else:
            idx_other = self.edit_role.findText("Other")
            self.edit_role.setCurrentIndex(idx_other if idx_other >= 0 else 0)

        self.edit_affiliation.setText(user.get("affiliation") or "")
        self.edit_department.setText(user.get("department") or "")
        self.edit_phone.setText(user.get("phone") or "")
        self.edit_website.setText(user.get("website") or "")
        
        # Admin flag
        is_admin = user.get("is_admin") == 1
        self.edit_is_admin.setChecked(is_admin)
        self.edit_allow_autologin.setChecked(bool(user.get("allow_passwordless_login")))
        
        # Reset staged password
        self.temp_password = None

        # Permission checks for enabling fields
        active_id = cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")
        current_user_data = next((u for u in self.users if u["user_id"] == active_id), None)
        is_current_admin = current_user_data.get("is_admin") == 1 if current_user_data else False
        
        has_any_admin = any(u.get("is_admin") == 1 for u in self.users)
        
        self.edit_is_admin.setEnabled(not has_any_admin or is_current_admin)
        self._autologin_permitted = (
            is_current_admin or self.selected_user_id == active_id or not has_any_admin
        )
        self._apply_admin_autologin_rule()

        can_change_pw = is_current_admin or (self.selected_user_id == active_id)
        self.btn_change_password.setEnabled(can_change_pw)

        self.edit_address.setText(user.get("address") or "")
        self.edit_details.setPlainText(user["details"] or "")

        self.btn_set_active.setEnabled(True)

    def _apply_admin_autologin_rule(self, *_):
        """Keep the autologin checkbox consistent with the admin flag.

        Admin accounts can never use passwordless login (enforced server-side),
        so when "Is Administrator" is checked the autologin checkbox is forced
        off and disabled. Otherwise it follows the per-user permission flag.
        """
        is_admin = self.edit_is_admin.isChecked()
        if is_admin:
            self.edit_allow_autologin.setChecked(False)
            self.edit_allow_autologin.setEnabled(False)
            self.edit_allow_autologin.setToolTip(
                "Admin accounts cannot use passwordless login."
            )
        else:
            self.edit_allow_autologin.setEnabled(getattr(self, "_autologin_permitted", True))
            self.edit_allow_autologin.setToolTip(
                "Allow this user to obtain a login session without entering a password."
            )

    def on_new_user_clicked(self):
        """Prepare inputs for creating a new user with a generated UUID."""
        self.table.clearSelection()
        self.clear_details()
        self.is_creating_new = True

        new_uuid = str(uuid.uuid4())
        self.edit_user_uuid.setText(new_uuid)
        self.edit_user_id.setReadOnly(False)
        self.edit_user_id.setStyleSheet("")
        self.edit_user_id.setFocus()
        self.btn_set_active.setEnabled(False)
        
        # Check permissions for is_admin for new user creation
        active_id = cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")
        current_user_data = next((u for u in self.users if u["user_id"] == active_id), None)
        is_current_admin = current_user_data.get("is_admin") == 1 if current_user_data else False
        has_any_admin = any(u.get("is_admin") == 1 for u in self.users)
        
        self.edit_is_admin.setEnabled(not has_any_admin or is_current_admin)
        self.edit_allow_autologin.setChecked(False)
        self._autologin_permitted = not has_any_admin or is_current_admin
        self._apply_admin_autologin_rule()
        self.btn_change_password.setEnabled(True)
        self.temp_password = None
        
        self.status_bar.showMessage("Configuring new user. Complete fields and click 'Save Changes'.")

    def on_change_password_clicked(self):
        if not self.selected_user_id and not self.is_creating_new:
            QMessageBox.warning(self, "Selection Required", "Please select or create a user first.")
            return
            
        user_id = self.selected_user_id or self.edit_user_id.text().strip()
        is_admin = self.edit_is_admin.isChecked()
        
        dlg = PasswordChangeDialog(user_id=user_id, is_admin=is_admin, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.temp_password = dlg.password
            self.status_bar.showMessage("Password staged. Click 'Save Changes' to apply.")

    def on_save_clicked(self):
        """Save a new user or update an existing user in the database."""
        user_uuid = self.edit_user_uuid.text().strip()
        user_id = self.edit_user_id.text().strip()
        display_name = self.edit_display_name.text().strip()
        email = self.edit_email.text().strip()
        role = self.edit_role.currentText()
        affiliation = self.edit_affiliation.text().strip()
        department = self.edit_department.text().strip()
        phone = self.edit_phone.text().strip()
        website = self.edit_website.text().strip()
        address = self.edit_address.toPlainText().strip()
        details = self.edit_details.toPlainText().strip()
        is_admin = 1 if self.edit_is_admin.isChecked() else 0
        allow_autologin = 1 if self.edit_allow_autologin.isChecked() else 0
        
        password = self.temp_password

        if not user_id:
            QMessageBox.warning(self, "Validation Error", "User ID is required.")
            return
        if not display_name:
            QMessageBox.warning(self, "Validation Error", "Display Name is required.")
            return
        if email:
            if "@" not in email or "." not in email.split("@")[-1]:
                QMessageBox.warning(self, "Validation Error", f"Invalid email format: '{email}'")
                return

        active_id = cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")

        try:
            client = self.make_mfdb_client()
            has_token = bool(getattr(client, "token", None))
            old_user_id = self.selected_user_id if not self.is_creating_new else None
            logging.info(
                "User Editor: saving user '%s' via MFDB RPC (old_user_id=%s, auth_token=%s, allow_autologin=%s)",
                user_id,
                old_user_id,
                has_token,
                bool(allow_autologin),
            )
            user_data = {
                "user_uuid": user_uuid or None,
                "user_id": user_id,
                "display_name": display_name,
                "email": email or None,
                "role": role,
                "affiliation": affiliation or None,
                "department": department or None,
                "phone": phone or None,
                "website": website or None,
                "address": address or None,
                "details": details or None,
                "is_admin": is_admin,
                "allow_passwordless_login": allow_autologin,
                "requester_id": active_id
            }
            if old_user_id and old_user_id != user_id:
                user_data["old_user_id"] = old_user_id
            if password is not None:
                user_data["password"] = password

            client.save_user(user_data)
            if old_user_id and old_user_id != user_id:
                self.rename_local_user_state(old_user_id, user_id)
            logging.info("User Editor: saved user '%s'", user_id)
            if allow_autologin == 0:
                self.disable_local_autologin(user_id)
            self.temp_password = None
            self.status_bar.showMessage(f"Successfully saved user: {display_name}")
            self.load_users(select_user_id=user_id)
        except Exception as e:
            logging.exception("User Editor: failed to save user '%s' via MFDB RPC", user_id)
            QMessageBox.critical(self, "ZMQ RPC Error", f"Could not save user:\n{e}")

    def rename_local_user_state(self, old_user_id: str, new_user_id: str) -> None:
        """Update local settings and stored tokens after a username rename.

        Parameters
        ----------
        old_user_id : str
            Previous username.
        new_user_id : str
            New username.
        """
        from chisurf.core.mfdb.security.credentials import (
            rename_runtime_session_token,
            rename_session_token,
        )
        from chisurf.core.settings.settings_utils import set_mfdb_login_settings

        mfdb_settings = cs_settings.cs_settings.setdefault("mfdb", {})
        server_host = mfdb_settings.get("last_server", "127.0.0.1")
        server_port = int(mfdb_settings.get("last_port", 8765))
        rename_runtime_session_token(server_host, server_port, old_user_id, new_user_id)
        rename_session_token(server_host, server_port, old_user_id, new_user_id)
        if mfdb_settings.get("default_user_id") == old_user_id:
            mfdb_settings["default_user_id"] = new_user_id
            os.environ["MFDB_DEFAULT_USER_ID"] = new_user_id
            if hasattr(cs_settings, "mfdb"):
                cs_settings.mfdb["default_user_id"] = new_user_id
            set_mfdb_login_settings(mfdb_settings)
        logging.info("User Editor: renamed local user state from '%s' to '%s'", old_user_id, new_user_id)

    def make_mfdb_client(self):
        """Create an MFDB client using the current session token when available.

        Returns
        -------
        MFDBClient
            Client configured with the current user's runtime or stored token.
        """
        from chisurf.core.mfdb.security.credentials import (
            load_runtime_session_token,
            load_session_token,
            store_runtime_session_token,
        )
        from mfdb.admin.gui.client import MFDBClient

        mfdb_settings = cs_settings.cs_settings.get("mfdb", {})
        server_host = mfdb_settings.get("last_server", "127.0.0.1")
        server_port = int(mfdb_settings.get("last_port", 8765))
        user_id = mfdb_settings.get("default_user_id", "user_default")
        client = MFDBClient(host=server_host, cmd_port=server_port, pub_port=server_port + 1)
        token = load_runtime_session_token(server_host, server_port, user_id)
        if token is None:
            token = load_session_token(server_host, server_port, user_id)
        if token:
            client.token = token
            store_runtime_session_token(server_host, server_port, user_id, token)
        logging.info(
            "User Editor: created MFDB client for %s:%s as '%s' (auth_token=%s)",
            server_host,
            server_port,
            user_id,
            bool(token),
        )
        return client

    def disable_local_autologin(self, user_id: str) -> None:
        """Disable saved local autologin credentials for a user.

        Parameters
        ----------
        user_id : str
            MFDB user whose local autologin state should be cleared.
        """
        from chisurf.core.mfdb.security.credentials import delete_runtime_session_token, delete_session_token
        from chisurf.core.settings.settings_utils import set_mfdb_login_settings

        mfdb_settings = cs_settings.cs_settings.setdefault("mfdb", {})
        server_host = mfdb_settings.get("last_server", "127.0.0.1")
        server_port = int(mfdb_settings.get("last_port", 8765))
        delete_session_token(server_host, server_port, user_id)
        delete_runtime_session_token(server_host, server_port, user_id)
        logging.info("User Editor: cleared local autologin tokens for user '%s'", user_id)

        if mfdb_settings.get("default_user_id") == user_id:
            mfdb_settings["autologin"] = False
            if hasattr(cs_settings, "mfdb"):
                cs_settings.mfdb["autologin"] = False
            if not set_mfdb_login_settings(mfdb_settings):
                self.status_bar.showMessage("Autologin disabled, but settings could not be written.")

    def on_set_active_clicked(self):
        """Set the selected user as the default active user in the settings."""
        if not self.selected_user_id:
            return

        # Update in-memory settings
        if "mfdb" not in cs_settings.cs_settings:
            cs_settings.cs_settings["mfdb"] = {}
        cs_settings.cs_settings["mfdb"]["default_user_id"] = self.selected_user_id
        os.environ["MFDB_DEFAULT_USER_ID"] = self.selected_user_id

        # Update the properties loaded in the package scope
        if hasattr(cs_settings, "mfdb"):
            cs_settings.mfdb["default_user_id"] = self.selected_user_id

        # Write to settings_chisurf.yaml
        try:
            with open(cs_settings.chisurf_settings_file, "w", encoding="utf-8") as f:
                yaml.dump(cs_settings.cs_settings, f, default_flow_style=False)
            QMessageBox.information(
                self, "Active User Changed",
                f"Active user successfully changed to '{self.selected_user_id}'."
            )
            self.load_users()
        except Exception as e:
            QMessageBox.critical(
                self, "Settings Error",
                f"Could not persist active user to configuration file:\n{e}"
            )

    def on_delete_user_clicked(self):
        """Safely delete a user if they have not committed any data."""
        if not self.selected_user_id:
            QMessageBox.warning(self, "Selection Required", "Please select a user to delete.")
            return

        user_id = self.selected_user_id

        # 1. user_default can never be deleted
        if user_id == "user_default":
            QMessageBox.warning(
                self, "Action Prohibited",
                "The default user ('user_default') is required by the system and cannot be deleted."
            )
            return

        # 2. Currently active default user cannot be deleted
        active_id = cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")
        if user_id == active_id:
            QMessageBox.warning(
                self, "Action Prohibited",
                "The selected user is currently configured as the active user.\n"
                "Please select and switch to another active user before deleting this one."
            )
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete user '{user_id}'?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                client = self.make_mfdb_client()
                logging.info("User Editor: deleting user '%s' via MFDB RPC", user_id)
                client.delete_user(user_id, force=False)
                logging.info("User Editor: deleted user '%s'", user_id)
                self.status_bar.showMessage(f"Successfully deleted user: {user_id}")
                self.load_users()
            except Exception as e:
                logging.exception("User Editor: failed to delete user '%s' via MFDB RPC", user_id)
                # The backend raises a ValueError if deletion is prohibited (e.g., committed data)
                current_user_data = next((u for u in self.users if u["user_id"] == active_id), None)
                is_current_admin = current_user_data.get("is_admin") == 1 if current_user_data else False
                
                if is_current_admin:
                    override_reply = QMessageBox.question(
                        self, "Admin Override",
                        f"Could not delete user: {e}\n\nDo you want to FORCE delete this user? This will delete the user but leave their committed data intact in the database.",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    if override_reply == QMessageBox.Yes:
                        try:
                            logging.warning("User Editor: force-deleting user '%s' via MFDB RPC", user_id)
                            client.delete_user(user_id, force=True, requester_id=active_id)
                            logging.info("User Editor: force-deleted user '%s'", user_id)
                            self.status_bar.showMessage(f"Successfully force-deleted user: {user_id}")
                            self.load_users()
                        except Exception as force_e:
                            logging.exception("User Editor: force delete failed for user '%s'", user_id)
                            QMessageBox.critical(self, "Action Prohibited", f"Force deletion failed:\n{force_e}")
                else:
                    QMessageBox.critical(self, "Action Prohibited", f"Could not delete user:\n{e}")
