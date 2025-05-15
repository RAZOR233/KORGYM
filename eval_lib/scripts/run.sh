

#!/bin/bash
game_name=34-one_touch_drawing
game_host=0.0.0.0
game_port=8775

cd ../../game_lib/$game_name
nohup python game_lib.py --host $game_host -p $game_port > game_server.out 2>&1 &
game_pid=$!
echo "runing Game at process: $game_pid"
sleep 2s
cd ../..
python -m eval_lib.eval -o results -m model_name -a base_url -k api_key -g $game_name -u http://localhost:$game_port -l 4
kill $game_pid