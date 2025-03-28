#!/bin/bash
cd /workspaces/streamlitstogo/
git add --all
git commit -m "autoCommit $(date +'%Y%m%d.%H%M%S')"
git push
exit
