The files were created using ``src/create_dataset_csv.ipynb``  

## Dataset Types
* default: contains base_prompt and source_prompt
* divided: the prompts are divided at each intervention location

## Source Number Types
* default
  * base_num_1, base_num_2, source_num_1, and source_num_2 are random numbers
  * base_num_2 and source_num_2 do not contain 0's
* h2: same as default, except
  * source_num_1 is the same as base_num_1
  * source_num_2 differs from base_num_1 only in the hundreds digit
* h : same as h2, except
  * base_sum and source_sum < 1000

## Truncation
* default: right before the final output
* pre_result: before the model presents the result of its reasoning within the reaoning tokens
* pre_sum: before the model calculates the final sum (often the same place as pre_result)

