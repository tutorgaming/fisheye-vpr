class DatabaseGenerator(object):
    def __init__(self, global_desc_extractor, db_dataset):
        # Global Description Extractor
        self.global_desc_extractor = global_desc_extractor
        # Database for searching
        self.db_dataset = db_dataset

    def create_database(
        self
    )-> dict:
        """
        Given the pytorch's Dataset
        Create the database of Global Description extracted from the dataset
        """
        result = {}
        # Iterate through the dataset
        for idx in range(len(self.db_dataset)):
            # Get the Image
            image = self.db_dataset[idx]
            # Extract the Global Description
            global_desc = self.global_desc_extractor.extract(image)
            # Add to the Database
            result[idx] = global_desc
        return result