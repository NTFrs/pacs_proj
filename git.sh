file="$1"
message="$2"
git add $file
git commit -am $message
git push
