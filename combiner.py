import os


def combine_steps_to_txt(output_file="combined_steps.txt"):
    steps_dir = "steps"
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(steps_dir):
            if filename.endswith(".py"):
                filepath = os.path.join(steps_dir, filename)
                with open(filepath, "r", encoding="utf-8") as infile:
                    outfile.write(f"# {filename}\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")


if __name__ == "__main__":
    combine_steps_to_txt()
