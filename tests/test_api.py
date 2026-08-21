from fastapi.testclient import TestClient
from augagent.api import app
from typer.testing import CliRunner
from augagent.cli import app as cli_app

client = TestClient(app)

def test_api_kickoff_validation():
    """Test that the /kickoff endpoint returns 401 on missing auth."""
    response = client.post("/kickoff", json={})
    assert response.status_code == 401

def test_cli_help():
    """Test that the CLI application loads and shows help."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "AugAgent Command Line Interface" in result.stdout
