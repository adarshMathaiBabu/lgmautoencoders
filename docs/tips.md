# Tips & Tricks

**NOTE:**
Most of the Jupyter-related tricks here **do not apply if you use Jupyter through VSCode**.
Usually, these kernels will be killed automatically if you lose connection to the compute node!
The last section ("More Notebook Tips") should still be relevant, however.

## tmux: Persistent Jupyter & Other Processes

One annoying thing is that, if you log out or lose connection, your running processes will die.
That includes Jupyter servers, as well as any model training scripts, for example.
You can run them in a "detached" window to keep them persistent.
Common tools for this are `screen` or `tmux`. 
We recommend `tmux`.
There are some rather involved tutorials around the web, for example [LINK](https://github.com/tmux/tmux/wiki).
The main things you should focus on are:
- How to create a new session: `tmux new -s name` where `name` is the desired session name.
- How to attach to that session: `tmux attach -t name`.
- Some of the common keyboard shortcuts, e.g. how to detach from a session without killing it, or
how to open multiple windows and navigate between them.
    - All shortcuts consist of `ctrl + b`, followed by a single key (do not hold `ctrl` here!).
    Basically, `crtl + b` puts you into "tmux command mode", and the next key gives the command.
    - `d` to detach and get back to the main shell.
    Be sure not to hit `ctrl + d`, as this will kill your session/window.
    - `c` to create a new window.
    - `n` or `p` to go the next/previous window.
    - `[` to enter scrolling mode. You can now scroll the window using arrow keys or mouse wheel.
    You cannot scroll back in tmux otherwise. Hit `ctrl + c` to stop scrolling.

If you start your Jupyter server inside a tmux session, it should persist even when you disconnect.
This means you can pick up your computations right where you left off by simply attaching to the
session after reconnecting.
NOTE that PyTorch reserves GPU memory for as long as a process persists.
Be sure to kill old kernels you no longer need to free up GPU memory.

You can also run your training processes using Lightning CLI in a tmux session.
This lets you start a process, leave the computer, and come back later to the finished model.
Of course you should make sure everything works fine, first.


## Long-term Maintenance

A big thing to watch out for with Jupyter servers is that the "kernels" (notebook persistent states)
will keep running for as long as the server is running.
This means that, if you are running your Jupyter server inside `tmux`, the notebooks will never be
killed, and the processes will run forever.
This can be an issue, as Pytorch will keep reserving GPU memory.
In the worst case, your old processes will clog up all available memory, and at some point you may
not be able to run anything.
Thus, be sure to **shut down old kernels** regularly, or completely restart the Jupyter server.

### Handling Long-Running Processes in Notebooks
One advantage of the previous point is that you can run stuff in a notebook today, come back
tomorrow, and pick up right where you left!
This also means you could start running a model over night, and come back the next day.
However, there are some caveats to this.

Basically, if you **leave the browser tab open** and disconnect from the server, whenever you come
back and reconnect, all the outputs will be flushed to the notebook (usually with some delay).
This can be a bit spammy, and Jupyter has limits on the output rate.
Sometimes this will result in outputs "pausing" and many being skipped, which may result in a loss
of information about whatever processes you were running.
This can especially happen with things like `tqdm` progress bars, as these send output very
frequently.

However, the following will "break the transmission":
- Closing the browser tab where the notebook is running.
- Closing the browser entirely while the tab is still open (reopening and restoring the previous
session does _not_ work).
- Shutting down your computer. Standby is okay.

If you do any of those things, once you reconnect to the notebook, you will no longer get outputs
from any processes that are still running.
They _will_ run fine, however, just "in the background".
Once whatever cell was running is finished (you can also interrupt execution as usual), you can run
other cells and will once again receive outputs.

**tl;dr:** If you want to run long processes entirely in the notebook and want to keep getting
outputs like prints or images, you must not close the browser tab or shut down your computer while
the process is still running.
We mention once again that this does not apply to running Jupyter from within VSCode.

If you ever do close the tab, just going to `localhost:xxxx` in your browser is likely _not_ enough;
you will need to get the token again.
One fun way to do this is to go to where the server command is running (e.g. `tmux` session), hit
`ctrl+c`, which will prompt in a message containing the full address _with_ the token, and ask if
you want to kill the process.
Don't press anything, and it will just resume after a couple seconds.
Then you can copy & paste the token from this message.


## More Notebook Tips

Working in notebooks can be convenient, but they have some peculiarities you sometimes have to work
around.
Here are some more general tips.

### Save Your Work!
Notebooks autosave sometimes, but not very frequently.
Make sure you are regularly saving your notebooks, either via "File -> Save Notebook" or the
associated keyboard shortcut (CTRL/CMD + S).
In particular, **do this before downloading the notebook for submission**, as otherwise your outputs
may not be saved.
Note that saving can take a moment for large notebooks (e.g. with many image outputs).
A notebook with unsaved changes usually has a little dot symbol in the browser/lab tab.

### Oh God I Lost Everything
When manually saving, Jupyter likes to create _checkpoints_ in the `.ipynb_checkpoints` folder.
In the notebook, you can select "File -> Revert Notebook to Checkpoint" in case you lost some work 
somehow.
This _may_ restore what you did earlier.

### Reloading Imports
Working with imports from other `.py` files can be annoying, since once imported, Python will not
re-import them even if the files were changed.
This can slow down development:
Each time you make a change in a `.py` file, you will have to reset your notebook and import
everything again, losing all progress.
You can force Jupyter to re-import functions each time they are accessed via the following:

```
%load_ext autoreload
%autoreload 2
```

Put these lines at the top of your notebook, before importing anything. Some notes:
- You might want to _not_ use this feature once your code is finished, as the constant re-importing
could slow down your program.
- I have witnessed some cases where this possibly lead to issues and the program breaking (though
there is a chance that this was related to something else).

If you run into unexplainable errors with code that should be working, try turning this feature off.
