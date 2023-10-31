import os
import shutil
import pathlib
from datetime import datetime
from termcolor import colored
import click

@click.group()
def main() -> None:
    pass


@main.command()
@click.argument('keywords', type=str, required=1)
def init(keywords):
    """
    This function takes in a set of keywords that are used alongside the
    researchers name and current date to form a unique analysis directory
    name / id. It creates the directory, initializes pdpp, and creates
    a config.yaml file that can be used to execute the project pipeline. 
    """
    who = os.getlogin()
    when = datetime.now().strftime("%Y%m%d")
    analysis_id = f'{who}_{when}_{keywords}'.lower().replace(' ', '_')
    analysis_dir = os.path.join('analyses/', analysis_id)
    if os.path.exists(analysis_dir) is False:
        os.mkdir(analysis_dir)
        os.mkdir(os.path.join(analysis_dir, 'pipeline_logs'))
        os.system(f'cd {analysis_dir} && pdpp init')
        config = f'{analysis_dir}/config.yaml'
        with open(config, 'w+') as f:
            f.write(f'analysis_id: {analysis_id}\n')
            f.write('subreddits:\n')
            f.write('sample_n_conversations:\n')
        readme = f'{analysis_dir}/README.md'
        if os.path.exists(readme) is False:
            with open(readme, 'w+') as f:
                f.write(f'# {analysis_id}\n\n')
                f.write(f'## Data Origin\n\n')
                f.write('The datasets used in this analysis were produced by the following automated pipeline:\n')
                f.write('\n![](dependencies_sparse.png)\n\n')
                f.write('\n![](dependencies_all.png)\n\n')
                f.write('> [!info] pdpp graph is not automated!\n')
                f.write('> To ensure the dependency graphs are up to date, run.\n')
                f.write('> ```bash\n')
                f.write('> cd pipeline && pdpp graph\n')
                f.write(f'> cp pipeline/dependencies_*.png {analysis_dir}\n')
                f.write('> ```\n')
        click.echo(colored(f'🔥 Initialized analysis directory {analysis_dir}.', 'red'))
    else:
        click.echo(colored(f'👎 {analysis_dir} already exists. Choose other keywords.', 'red'))
        

@main.command()
@click.argument('analysis_dir', type=click.Path(exists=True), required=1)
def run(analysis_dir):
    """
    This function accepts a path (e.g., analyses/johnmclevey_20231029_yo_yo_yo),
    copies the config file into the _import_ folder of the project pipeline, 
    executes the project pipeline, copies the contents of the _export_ directory
    of the pipeline into the _import_ analysis of the analysis directory, and then 
    deletes the config file from the pipeline _import_ directory. 

    Note that adding this file (with the specific file name) to the pipeline _import_
    directory will ALWAYS trigger the full project pipeline, as everything is downstream 
    of _import_/config.yaml. This is intentional!
    """
    config_analysis = f'{analysis_dir}/config.yaml'
    config_pipeline = 'pipeline/_import_/config.yaml'
    shutil.copy(config_analysis, config_pipeline)
    os.system('cd pipeline && pdpp run')
    
    shutil.copytree('pipeline/_export_', 
        f'{analysis_dir}/_import_/results/', 
        dirs_exist_ok=True)
    
    logfiles = list(pathlib.Path('pipeline').rglob('*.log'))
    for log in logfiles:
        shutil.copy(log, f'{analysis_dir}/pipeline_logs')
        click.echo(colored(f'Found {log}.', 'blue'))

    dependencies = list(pathlib.Path('pipeline').rglob('dependencies_*.png'))
    if len(dependencies) > 0:
        for graph in dependencies:
            shutil.copy(graph, f'{analysis_dir}/')
            click.echo(colored(f'Found {graph}, may be outdated.', 'blue'))

    os.remove(config_pipeline)
