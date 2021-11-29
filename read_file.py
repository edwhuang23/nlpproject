import h5py
filename = "TRAAAAW128F429D538.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys:", str(f.keys()))
    analysis = list(f.keys())[0]

    # Get the data keys
    data_key = list(f[analysis])
    data = f['analysis']
    
    for key in data_key:
        print("Key:", str(key))
        print(data[key][()])


