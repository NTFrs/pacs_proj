file="$1"
message="$2"
git pull
git add $file
git commit -am $message
git push
