import pathlib
import uuid

import pytest

from chisurf.core.mfdb.repository import MFDatabase


def test_default_branch_setup(tmp_path: pathlib.Path) -> None:
    """Verify a new database has main branch defaults."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        # Verify default 'main' branch exists
        main_branch = db.get_branch("main")
        assert main_branch is not None
        assert main_branch["name"] == "main"
        assert main_branch["branch_uuid"] == "00000000-0000-0000-0000-000000000000"
        assert main_branch["deleted_at"] is None

        # Verify list_branches contains main
        branches = db.list_branches()
        assert len(branches) == 1
        assert branches[0]["name"] == "main"

        # Default user should have active branch pointing to main
        active = db.get_user_active_branch("user_default")
        assert active is not None
        assert active["name"] == "main"

def test_create_and_get_branch(tmp_path: pathlib.Path) -> None:
    """Verify branches can be created and fetched by UUID or name."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        # Create a new branch
        branch_uuid = str(uuid.uuid4())
        created_uuid = db.create_branch(
            branch_uuid=branch_uuid,
            name="experiment_a",
            description="Branch for experiment A",
        )
        assert created_uuid == branch_uuid

        # Get by uuid
        b1 = db.get_branch(branch_uuid)
        assert b1 is not None
        assert b1["name"] == "experiment_a"
        assert b1["description"] == "Branch for experiment A"

        # Get by name
        b2 = db.get_branch("experiment_a")
        assert b2 is not None
        assert b2["branch_uuid"] == branch_uuid

        # Duplicate branch name should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            db.create_branch(name="experiment_a")

def test_delete_branch_constraints(tmp_path: pathlib.Path) -> None:
    """Verify protected and active branches cannot be deleted."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        # Cannot delete main branch
        with pytest.raises(ValueError, match="Cannot delete the main branch"):
            db.delete_branch("00000000-0000-0000-0000-000000000000")

        # Create a branch
        branch_uuid = db.create_branch(name="experiment_b")

        # Set it active for default user
        db.set_user_active_branch("user_default", branch_uuid)

        # Cannot delete active branch
        with pytest.raises(ValueError, match="currently the active branch"):
            db.delete_branch(branch_uuid)

        # Switch back to main
        db.set_user_active_branch("user_default", "00000000-0000-0000-0000-000000000000")

        # Now can delete
        db.delete_branch(branch_uuid)
        assert db.get_branch(branch_uuid) is None

def test_operation_advances_head(tmp_path: pathlib.Path) -> None:
    """Verify operations advance only the active branch head."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        # Default user is active on main
        # Record a new operation
        db.record_operation(
            operation_id="op_1",
            operation_type="import",
            status="success",
            operator_user_id="user_default",
        )

        # Main head should now point to op_1
        main_branch = db.get_branch("main")
        assert main_branch["head_operation_id"] == "op_1"

        # Create new branch from main's head
        feature_branch_uuid = db.create_branch(
            name="feature_branch",
            parent_branch_uuid=main_branch["branch_uuid"],
            head_operation_id=main_branch["head_operation_id"],
        )

        # Set user's active branch to the new branch
        db.set_user_active_branch("user_default", feature_branch_uuid)

        # Record another operation
        db.record_operation(
            operation_id="op_2",
            operation_type="fitting",
            status="success",
            operator_user_id="user_default",
        )

        # Main head should still be op_1
        main_branch = db.get_branch("main")
        assert main_branch["head_operation_id"] == "op_1"

        # Feature branch head should now point to op_2
        feat_branch = db.get_branch(feature_branch_uuid)
        assert feat_branch["head_operation_id"] == "op_2"

def test_fork_branch_from_historical_operation(tmp_path: pathlib.Path) -> None:
    """Verify branch forking can start from an older operation."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        db.record_operation(
            operation_id="op_1",
            operation_type="import",
            status="success",
            operator_user_id="user_default",
        )
        db.record_operation(
            operation_id="op_2",
            operation_type="fitting",
            status="success",
            operator_user_id="user_default",
        )

        fork_uuid = db.fork_branch(
            source_branch_uuid="00000000-0000-0000-0000-000000000000",
            name="parallel_from_op_1",
            head_operation_id="op_1",
            created_by_user_id="user_default",
        )

        fork = db.get_branch(fork_uuid)
        assert fork["parent_branch_uuid"] == "00000000-0000-0000-0000-000000000000"
        assert fork["head_operation_id"] == "op_1"
        assert db.get_branch("main")["head_operation_id"] == "op_2"

def test_jump_user_to_operation_creates_active_parallel_branch(tmp_path: pathlib.Path) -> None:
    """Verify user time travel creates and activates a parallel branch."""
    db_path = tmp_path / "test_branching.db"
    with MFDatabase(db_path) as db:
        db.record_operation(
            operation_id="op_1",
            operation_type="import",
            status="success",
            operator_user_id="user_default",
        )
        db.record_operation(
            operation_id="op_2",
            operation_type="fitting",
            status="success",
            operator_user_id="user_default",
        )

        branch = db.jump_user_to_operation(
            user_id="user_default",
            operation_id="op_1",
            branch_name="default_at_op_1",
        )

        assert branch["name"] == "default_at_op_1"
        assert branch["head_operation_id"] == "op_1"
        active = db.get_user_active_branch("user_default")
        assert active["branch_uuid"] == branch["branch_uuid"]

        db.record_operation(
            operation_id="op_3",
            operation_type="fitting",
            status="success",
            operator_user_id="user_default",
        )
        assert db.get_branch(branch["branch_uuid"])["head_operation_id"] == "op_3"
        assert db.get_branch("main")["head_operation_id"] == "op_2"
