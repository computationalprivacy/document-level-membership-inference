"""Main file to split in chunks."""
import configargparse
import pickle
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import random 

def get_parser():
    """Return parser for the args."""
    parser = configargparse.ArgParser()
    parser.add_argument(
        '-c', '--config', required=False, is_config_file=True,
        help='Config file path')

    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--path_to_member_data", type=str, required=True)
    parser.add_argument("--path_to_non_member_data", type=str, required=True)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--filter_on_date", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--n_pos_chunk", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    return parser

def filter_on_date(dataset, date_name, min_date=1850, max_date=1910):
    all_indices = range(len(dataset))
    valid_indices = []

    if len(all_indices) > 10000:
        all_indices = random.sample(all_indices, 10000)
    
    for i in tqdm(all_indices):
        date = dataset[i][date_name]
        if isinstance(date, str) or isinstance(date, int):
            if int(date) >= min_date and int(date) <= max_date:
                valid_indices.append(i)
    
    sub_dataset = dataset.select(valid_indices)
    
    print("Number of documents before date filtering: ", len(dataset))
    print("Number of documents after date filtering: ", len(sub_dataset))
    
    return sub_dataset

def remove_small_docs(dataset, min_tokens):
    all_indices = range(len(dataset))
    all_relevant_indices = []

    if len(all_indices) > 20000:
        all_indices = random.sample(all_indices, 20000)
    
    for idx in tqdm(all_indices):
        if len(dataset[idx]['input_ids']) > min_tokens:
            all_relevant_indices.append(idx)

    sub_dataset = dataset.select(all_relevant_indices)

    print("Number of documents before min token filtering: ", len(dataset))
    print("Number of documents after min token filtering: ", len(sub_dataset))

    return sub_dataset

def main(args):
    """Main function to call."""

    random.seed(args.seed)
    
    print("Loading the raw data..")
    og_member_dataset = load_from_disk(args.path_to_member_data)
    if args.filter_on_date:
        og_member_dataset = filter_on_date(og_member_dataset, date_name="publication_date")
    og_member_dataset = og_member_dataset.select_columns(['input_ids', 'attention_mask'])
    og_non_member_dataset = load_from_disk(args.path_to_non_member_data)
    if args.filter_on_date:
        og_non_member_dataset = filter_on_date(og_non_member_dataset, date_name="original_publication")
    og_non_member_dataset = og_non_member_dataset.select_columns(['input_ids', 'attention_mask'])

    if args.min_tokens > 0:
        print("Removing the docs with limited tokens..")
        member_dataset = remove_small_docs(og_member_dataset, args.min_tokens)
        non_member_dataset = remove_small_docs(og_non_member_dataset, args.min_tokens)
    else:
        member_dataset = og_member_dataset
        non_member_dataset = og_non_member_dataset

    print("Composing the chunks...")
    members_samples_idx = random.sample(range(len(member_dataset)), args.n_chunks * args.n_pos_chunk)
    non_members_samples_idx = random.sample(range(len(non_member_dataset)), args.n_chunks * args.n_pos_chunk)

    for chunk_id in range(args.n_chunks):
        chunk_members = member_dataset.select(members_samples_idx[chunk_id * args.n_pos_chunk : (chunk_id + 1) * args.n_pos_chunk])
        chunk_non_members = non_member_dataset.select(non_members_samples_idx[chunk_id * args.n_pos_chunk : (chunk_id + 1) * args.n_pos_chunk])
        chunk_dataset_all = concatenate_datasets([chunk_members, chunk_non_members])
        labels = [1] * len(chunk_members) + [0] * len(chunk_members)

        chunk_dataset_all.save_to_disk(f"{args.output_dir}/{args.prefix}_{chunk_id}_min_tokens{args.min_tokens}_seed{args.seed}")
        with open(f"{args.output_dir}/{args.prefix}_{chunk_id}_labels.pickle", 'wb') as f:
            pickle.dump(labels, f)

if __name__ == '__main__':
    main(get_parser().parse_args())