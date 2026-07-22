import logging
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values

# ----------------------------------------------------------------------------
# Debug utilities
def print_dict_mean(d, prefix="", important_keys=None):
    """
    Print dictionary values, showing means for most keys but full values for important keys.
    
    Args:
        d: Dictionary to print
        prefix: String prefix for nested keys
        important_keys: List of key names that should print full values instead of means
    """
    if important_keys is None:
        important_keys = []
    
    for key in d:
        full_key = f"{prefix}.{key}" if prefix else key
        value = d[key]
        
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            print_dict_mean(value, full_key, important_keys)
        else:
            # Check if this key should print full information
            should_print_full = (
                key in important_keys or 
                full_key in important_keys or
                any(important_key in full_key for important_key in important_keys)
            )
            
            if should_print_full:
                # Print full value for important keys
                print(f"{full_key}: {value}")
            else:
                # Print mean for regular keys
                # if hasattr(value, 'mean'):
                #     mean_val = value.mean()
                #     print(f"{full_key}: {mean_val}")
                # elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                #     # Handle iterables that don't have mean method
                #     try:
                #         mean_val = np.mean(value)
                #     except:
                #         mean_val = torch.mean(torch.tensor(value, dtype=torch.float32))
                #     print(f"{full_key}: {mean_val}")
                # else:
                # For non-iterable values, just print the value
                print(f"{full_key}: {value}")

def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")

# ----------------------------------------------------------------------------
# Quality of life utilities
def format_value(value):
    if isinstance(value, float):
        if abs(value) < 1e-2:
            return f"{value:.2e}"
        return f"{value:.2f}"
    return str(value)

def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    # Sort metrics by key name for consistent display
    for key in sorted(metrics.keys()):
        value = metrics[key]
        formatted_value = format_value(value)
        table.add_row(key, formatted_value)

    # Create a panel with the table
    panel = Panel(
        table,
        title="Metrics",
        expand=False,
        border_style="bold green",
    )

    # Print the panel
    rprint(panel)
# ----------------------------------------------------------------------------
