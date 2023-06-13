import subprocess


def runcmd(cmd: str, verbose: bool = False, *args, **kwargs):
    """_summary_
    Args:
        cmd str: the command
        verbose (bool, optional): _description_. Defaults to False.
    Information:
    Command taken from:
        https://www.scrapingbee.com/blog/python-wget/
    """

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
