import subprocess
import sys


def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    return result.stdout


def prepare_environment():
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        run_cmd('git clone https://github.com/emiliodelgadouy/tesis.git')
        run_cmd('mkdir utils')
        run_cmd('mv ./tesis/utils/* ./utils')
