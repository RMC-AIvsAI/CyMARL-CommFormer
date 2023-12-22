
def rearrange_dictionary(dictionary):
    
    dictionary = {key: [value[0]] + sorted(value[1:], key=lambda x: x[-1]) if key != 'Red' else value for key, value in dictionary.items()}




    return dictionary


# Example usage:
my_dictionary = {
    "Red": ["Monitor", "Remove User1", "Restore User1", "Analyse User1"],
    "Green": ["Monitor", "Remove User1", "Remove User2", "Restore User1", "Restore User2", "Analyse User1", "Analyse User2"],
    "Blue": ["Monitor", "Remove User1", "Remove User2", "Remove User3", "Restore User1", "Restore User2", "Restore User3", "Analyse User1", "Analyse User2", "Analyse User3"]
}

my_list = ["Remove User1", "Remove User2", "Remove User3", "Restore User1", "Restore User2", "Restore User3", "Analyse User1", "Analyse User2", "Analyse User3"]

new_dictionary = rearrange_dictionary(my_dictionary)

print(new_dictionary)

