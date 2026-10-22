# Setting Up Your Node

## Basics

You have to set up your Python environment.
Set up a virtual environment, see [LINK](https://docs.python.org/3/library/venv.html).
Note that you have to use the `python3` command.
The installed version is 3.12. 

Activate the environment.
Now you can install stuff.
Except that you can't; the default `/tmp` directory is too small for many libraries like torch.
So you have to first change that:

`export TMPDIR=/var/tmp`

After this, you should be able to `pip3 install torch` and all the other stuff you need.
We have a `requirements.txt` in the repository!
This will take some time; in particular, it has to download and install all the CUDA libraries.
If you get any errors during this process (especially regarding permissions), please let us know.
Also, if you ever get missing import errors when using our code, please tell us so that we can fix
this in the requirements file.

## Jupyter & Port Forwarding

You should also install Jupyter via pip.
Then, you can start a server, either `jupyter notebook` or `jupyter lab` (we recommend `lab` for a
better interface).
Unfortunately, the Jupyter server is then on the... server.
So you have to get it to your machine somehow.
For that, use port forwarding as described in CONNECT.
If you have the server running and forwarded the correct port, you should be able to access it from
your local machine via any internet browser.
From there, you can create notebooks, run them etc. just like on Google Colab.
The difference is that no-one will kick you from the GPU. :)
Also, your file storage is of course persistent.

To access whatever is running on a specific port, go to `localhost:xxxx` in any web browser, where
`xxxx` is the port number.
For example, the default port for Jupyter is `8888`, so that would be `localhost:8888`.
However, Jupyter actually requires a login token!
When you start `jupyter lab`, it should output a whole lot of stuff, among which you can find the
full URL you need to go to including the token.

NOTE: The port needs to be forwarded both from the compute node to the connect server, and then from
there to your local machine!
Setting up the ssh config takes care of this automatically.

NOTENOTE: In theory, there is an always-running Jupyter server that you can simply access via URL
`https://jhub.cs.ovgu.de/jupyter/<node_name>`, for example https://jhub.cs.ovgu.de/jupyter/n02-05
for node `n02-05`.
However, I personally have not been able to access these servers in a long time.
But it may be a user issue.
So you can try simply accessing that URL, which works without port forwarding.
However, you have to log in to the zone at least once to set up your user's home directory, else
this will not work.
Besides, this server will be using the system Python, and not your virtual environment, so it's
likely not a valid choice anyway.
Better create your own Jupyter server!

Finally, you can also use Jupyter from some IDEs like VSCode.
While this tends to have a much better interface, using this on a server that you only occasionally
connect to comes with its own issues.


## Fixing CUDA Incompatibilities

Depending on which torch version you install, you may run into issues due to the system CUDA version
being different from the one that comes bundled with Pytorch.
Specifically, we ran into this with the most recent torch version (2.14), leading to a crash when
trying to run certain operations on the GPU.
The actual error is something like

`RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`

An easy fix seems to be to hide the system CUDA.
You can do this by running `unset LD_LIBRARY_PATH` in the console.
Afterwards, `echo $LD_LIBRARY_PATH` should return an empty string.
Any processes started after this point should run fine.

Note that the environment variable gets _reset_ every time you disconnect from the server!
Thus, you would have to re-run the `unset` command each time you connect.
Alternatively, you can add it to your `.bashrc`, which should run it automatically when you connect.
Also, you will likely need to run the command separately in your `tmux` sessions (in fact, in each
*new window* you create within a session!), but here it should keep working for as long as the
session persists.
