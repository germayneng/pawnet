#!/bin/bash

echo "User: $USER"
echo "Host UID: $HOST_USER_ID"
echo "Host GID: $HOST_USER_GID"

if [ -z "${USER}" ]; then
  echo "We need USER to be set!"; exit 100
fi

# if both not set we do not need to do anything
if [ -z "${HOST_USER_ID}" -a -z "${HOST_USER_GID}" ]; then
    echo "Using default."
fi

# reset user_id to either new id or if empty old (still one of above
# might not be set)
USER_ID=${HOST_USER_ID:=$USER_ID}
USER_GID=${HOST_USER_GID:=$USER_GID}

LINE=$(grep -F "${USER}" /etc/passwd)
# replace all ':' with a space and create array
array=( ${LINE//:/ } )

# home is 5th element
USER_HOME=${array[4]}

sed -i -e "s/^${USER}:\([^:]*\):[0-9]*:[0-9]*/${USER}:\1:${USER_ID}:${USER_GID}/"  /etc/passwd
sed -i -e "s/^${USER}:\([^:]*\):[0-9]*/${USER}:\1:${USER_GID}/"  /etc/group

chown -R ${USER_ID}:${USER_GID} ${USER_HOME}

exec su - "${USER}"