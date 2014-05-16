#!/bin/sh
gnuplot << EOF
plot 'begin.gpl' title 'Begin'
pause mouse
do for [i=0:$1-2]{ plot 'step-'.($1-i).'.gpl' title 'Step '.($1-i)
pause $2
}
plot "end.gpl" title 'End'
pause mouse
EOF