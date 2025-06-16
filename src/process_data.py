from data.data_processor import DataProcessor

def main():
    print("[+] Processing dataset into binary files...")
    processor = DataProcessor()
    processor.prepare_dataset()
    print("[+] Data processing completed successfully!")

if __name__ == "__main__":
    main() 