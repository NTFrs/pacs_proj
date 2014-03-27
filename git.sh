file="$1"
message="$2"
git add $1
git commit -am $2
git push
