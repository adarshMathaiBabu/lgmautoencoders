# Connecting To Your Node

**NOTE:**
If you are having trouble connecting, please read _the entire thing_ first, and try the config-based
setup described towards the bottom.
Please only ask for help if that fails!

You have to connect via `ssh`.
This works on Linux, Mac, or WSL (Linux subsystem for Windows).
For native Windows, you could use PuTTY, for example.
Or perhaps it works out of the box on modern Windows machines -- I haven't tried in a while. :)
If you made it this far, you hopefully put your keys into that cloud document, so you already set up
key-based authentication.
If you need a reminder, there is a guide here: [LINK](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server)

It generally works like

`ssh <username>@<servername>`

Your username depends on your group as defined in the cloud document TODO
There is one user per node, called `exgr<N>` where `<N>` is your group number in the cloud document.
So for example, if in the document you are group 5, your username is `exgr5`.

You have to first connect to the login-server.
The address is `fcm.cs.uni-magdeburg.de`.
So: `ssh exgr5@fcm.cs.uni-magdeburg.de`, for example.
If you made it there, you then have to connect to your node.
Your node name is also in the list. E.g. group 5 has node `n02-05`.
So, from the login-server, `ssh` again:

`ssh exgr5@n02-05`, for example.

**NOTE:**
You may be able to log in to `fcm`, but then not be able to log in to the node from there, as it will ask you for a password, which you don't have.
Apparently, the fix is to use the `-A` option for ssh.
So:

`ssh -A exgr5@fcm.cs.uni-magdeburg.de`

If this also fails, please try the config method described below before asking or help.


## SSH Config Setup

On your local machine, you likely have a `.ssh` folder somewhere in your user directory (no idea how
this goes for Windows, sorry).
There should be a file called `config` in there.
If not, create it.
Just a file called `config` with no extensions.
So it should be `~/.ssh/config`, where `~` is your home directory.
Treat it as a simple text file.
There are various things to put in there that make your life easier.
For example, the below regularly "pings" the server to prevent disconnection.
This can otherwise happen if you are only working on notebooks and not using the terminal.

```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 2
```

You can also give aliases to hosts and set a default login name to make the ssh commands shorter.

```
Host fcm
    HostName fcm.cs.ovgu.de
    User <myname>
```

From now on, just using `ssh fcm` is equivalent to `ssh <myname>@fcm.cs.uni-magdeburg.de`. 
Unfortunately, this will likely lead to a failure to log in to the compute node afterwards -- it
will ask for a password.
See the discussion on the `-A` argument for ssh further above.
To my understanding, adding a `ForwardAgent` in the config *should* be the same as the `-A` flag, so
you could try that:

```
Host fcm
    HostName fcm.cs.ovgu.de
    ForwardAgent Yes
    User myname
```

You can even do multiple ssh jumps in one:

```
Host n02-05
    ProxyJump fcm
    ForwardAgent Yes
    User myname
```

Now, calling `ssh n02-05` (FROM YOUR LOCAL MACHINE) will ssh into the login server, and then
immediately into your node!
This means you only need to run `ssh` a single time.
Finally, you can forward ports like this:

```
Host n02-05_jhub
    HostName n02-05
    ProxyJump fcm
    User myname
    ForwardAgent Yes
    LocalForward 8888 localhost:8888
    LocalForward 8889 localhost:8889
    LocalForward 6006 localhost:6006
```

This will forward the ports 8888, 8889, 6006 to the respective ports on your local machine.
For example, Jupyter by default opens on port 8888. When you start the Jupyter server, it will give
you a link with an access key -- paste that in your local browser and you should see Jupyter!
6006 is the standard port for Tensorboard.
8889 etc can be used in case you are running multiple Jupyter servers in parallel for some reason.
Of course you could add more ports if you wish.

**NOTE:**
There may still be issues with the `ProxyJump` method and ssh-key-based log-in.
In that case, you likely want to just use the `ssh -A` method described earlier.
There is also an `-L` option to forward ports, and a `-J` option to do a "Jump".
So in principle, you can do all these things with a command line call and never have to set up a
config!
But in general, the config should be far more comfortable to use, and tends to give people fewer
problems.

If you need any further help, talk to us! 
