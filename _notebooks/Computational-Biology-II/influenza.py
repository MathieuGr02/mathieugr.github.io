class Sequence:
    accession: str
    description: str
    date: str
    country: str
    length: int
    sequence: str
    ambiguous: str

    def __init__(self, accession, description, date, country, length, sequence, ambigious):
        self.accession = accession
        self.description = description
        self.date = date
        self.country = country
        self.length = length
        self.sequence = sequence
        self.ambiguous = ambigious

def read_fasta(path: str) -> list[Sequence]:
    sequences = []
    with open(path, 'r') as file:
        data = "".join(file.readlines()).split(">")
        for entry in data:
            if entry == "":
                continue
            entry = entry.split("|")
            accession  = entry[0]
            description = entry[1]
            date = entry[2]
            split = entry[3].split("\n")
            country = split[0]
            sequence = "".join(split[1:])
            ambiguous = len([b for b in sequence if b not in ['A', 'T', 'C', 'G']])

            sequences.append(Sequence(accession, description, date, country, len(sequence), sequence, ambiguous))

    return sequences


if __name__ == '__main__':
    sequences = read_fasta('h3n2.fasta')
    print(len(sequences))