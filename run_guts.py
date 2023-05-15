import argparse
from generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./models/GUTS.bin", help="Path to a pretrained model")
parser.add_argument("--input_text", type=str, default=None, help="Paragraph as input, which should be simplified by the generator")
parser.add_argument("--top_p", type=float, default=0, help="p value for nucleus sampling")
parser.add_argument("--top_k", type=int, default=1, help="K value for top-K sampling") # default to top-k=1 is greedy decoding
parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to return")
parser.add_argument("--temperature", type=float, default=1, help="Temperature value")
args = parser.parse_args()

generator = Generator(max_input_length=180, max_output_length=300)
if len(args.model_path) > 0:
    generator.reload(args.model_path)
generator.eval()

input_text = args.input_text
if input_text == None:
    input_text = "Die Herrschaft Schalksburg war ein mittelalterliches Territorium auf dem Gebiet des heutigen Zollernalbkreises. Am Beispiel seiner Entstehung und Entwicklung lässt sich die Territoriums- und Herrschaftsbildung des niederen und höheren Adels in Südwestdeutschland exemplarisch ablesen. Im 15. Jahrhundert bildete die Herrschaft Schalksburg zusammen mit der Herrschaft Mühlheim die Herrschaft Zollern-Schalksburg. Trotz des Verkaufs an Württemberg im Jahr 1403, durch den das Territorium im neu geschaffenen Amt Balingen aufging, wurde die Erinnerung an die Herrschaft Schalksburg besonders von zollerischer Seite wachgehalten."

print("\nORIGINAL:\n", input_text)
print("\n")

if args.top_k == 1: 
    # greedy
    sample_size = 1
else:           
    # sampling
    sample_size = args.num_samples
    
outputs = generator.run(input_text, top_k=args.top_k, top_p=args.top_p, sample_size=sample_size, temperature=args.temperature)

for idx in range(len(outputs)):
    print("Simplification #%i: \n" % idx)
    print(outputs[idx])
    print('\n')


