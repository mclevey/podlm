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
    It also clears any old log file from the pipeline to ensure a clean run.
    """
    who = os.getlogin()
    when = datetime.now().strftime("%Y%m%d")
    analysis_id = f'{who}_{when}_{keywords}'.lower().replace(' ', '_')
    analysis_dir = os.path.join('analyses/', analysis_id)
    if os.path.exists(analysis_dir) is False:
        os.mkdir(analysis_dir)
        os.mkdir(os.path.join(analysis_dir, 'pipeline_dependencies'))
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
        
    # clear out any logfiles from the pipeline that were stored in previous runs
    logfiles = list(pathlib.Path('pipeline').rglob('*.log'))
    if len(logfiles) > 0:
        for log in logfiles:
            os.remove(log)
            click.echo(colored(f'🧼 Scrubbed old log files from pipeline/ in preparation for new analysis.', 'red'))
    
        

@main.command()
@click.argument('analysis_dir', type=click.Path(exists=True), required=1)
def run(analysis_dir):
    """
    This function accepts a path to an analysis directory (created by the init command),
    copies the config file into the _import_ folder of the project pipeline, 
    executes the project pipeline, copies the contents of the _export_ directory
    from pipeline into the _import_/results directory of analysis. It also files 
    log files and dependency graphs into pipeline_logs and pipeline_dependencies.
    It currently LEAVES THE CONFIG FILE in the _import_ directory of the pipeline
    because it plays better with pdpp, but the config file will be overwritten if 
    a new run is executed. 
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
        log_file_string = str(log).split('/')[-1]
        click.echo(colored(f'🧐 FILED LOG {log_file_string}', 'yellow'))

    click.echo(colored('🧐 Updating dependency graphs...'.upper(), 'red'))
    os.system("cd pipeline && pdpp graph --files 'png' --style 'default'")
    
    dependencies = list(pathlib.Path('pipeline').rglob('dependencies_*.png'))
    if len(dependencies) > 0:
        for graph in dependencies:
            shutil.copy(graph, f'{analysis_dir}/pipeline_dependencies')
            graph_file_string = str(graph).split('/')[-1]
            click.echo(colored(f'🧐 FILED DEPENDENCY GRAPH {graph_file_string}', 'yellow'))
    
    # os.remove(config_pipeline) 
    # seems like it's best to leave the last config file in _import_ to keep pdpp happy. 
