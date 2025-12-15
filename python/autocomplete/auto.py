from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

def main():

    # ------------------------------------
    # tensor object example:
    # ------------------------------------
    # tensor([[  464,  2003,   286,  9552,   318,   829,   481,   307,  1290,   422,
    #          10282,    13,  1081,   262,   995,   338,  1353, 18954,  2323,   329,
    #          23214,    11,   484,   389,   783,  9648,   284,  5529,   257,  5448,
    #           3265,    13,   383,  7794,   318, 10833,   262,  2732,   286, 19717,
    #            284,  3494, 27069, 20691,   287,  5318,  4536,   290,  3745,  3081]])

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model.generation_config.pad_token_id = tokenizer.eos_token_id

    print("")
    print("Using input from console -------------")
    print("")

    input_text = input("Enter your text: ")

    print("")
    print("")

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # amount of ids is equal to amount of words in the sentence
    max_length = input_ids.size(dim=input_ids.ndim-1) 

    output = model.generate(
            input_ids 
            , max_length=int(max_length*100*.065)        
            , attention_mask=torch.tensor
            (
                [[
                    n-n + random.randint(0,1) for n in range(max_length)
                ]]
            )    
            , do_sample=True
            )

    str_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Output generated from input ----------")
    print(str_output)
    print("--------------------------------------")


    print("")
    print("")

    # use first output to create second one

    print("")
    print("Using output generated from first output")
    print("")

    input_text = str_output # first output

    # amount of ids is equal to amount of words in the sentence
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    max_length = input_ids.size(dim=input_ids.ndim-1) 

    # generate second output
    output = model.generate(
            input_ids 
            , max_length=int(max_length*100*.065)
            , attention_mask=torch.tensor(
                [[
                    n-n + random.randint(0,1) for n in range(max_length)
                ]]
            )
            , do_sample=True
            )

    str_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Output generated from first output ---")
    print(str_output)
    print("--------------------------------------")

    print("")

if __name__ == "__main__":
    main()