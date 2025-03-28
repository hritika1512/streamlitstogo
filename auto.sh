#!/bin/bash
cd /workspaces/streamlitstogo/
/usr/bin/git add --all
/usr/bin/git commit -m "autoCommit $(date +'%Y%m%d.%H%M%S')"
/usr/bin/git push
exit
