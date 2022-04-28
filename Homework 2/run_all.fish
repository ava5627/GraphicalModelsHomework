#! /bin/fish

for file in (ls ./networks)
    for method in minfill mindegree
        echo ./networks/$file ./Ordering/$file.$method.order
        python3 ./main.py ./networks/$file $method 100 ./orders/$file.$method
        ulimit -v 2000000
        ulimit -t 300
        time ve_code/ve ./networks/$file ./orders/$file.$method.order
    end
end

