from click.testing import CliRunner

from her2_stopping_rule.__main__ import main


def test_dummy():
    runner = CliRunner()
    with runner.isolated_filesystem():
        test_input = ["World"]
        result = runner.invoke(main, test_input)
        assert result.exit_code == 0
