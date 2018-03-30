#!/bin/bash

s="a b c a b c"
echo "##a " ${s##*a}
echo "##b " ${s##*b}
echo "##c " ${s##*c}

echo "#a " ${s#*a}
echo "#b " ${s#*b}
echo "#c " ${s#*c}

echo "%a " ${s%a*}
echo "%b " ${s%b*}
echo "%c " ${s%c*}


echo "%%a " ${s%%a*}
echo "%%b " ${s%%b*}
echo "%%c " ${s%%c*}

