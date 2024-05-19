import os
import shutil
import time
import sys


class FileReformater:
    def __init__(self, folder: str, file: str = '*'):
        self.current_time = time.strptime(time.ctime())

        self.convert_to_markdown(folder, file)

        if file == '*':
            for md_file in os.listdir(f'_notebooks/{folder}'):

                images_folder = f'{md_file[:-6]}_files'
                print(images_folder)
                if os.path.isdir(f'_{folder}/{images_folder}'):
                    self.relocate_images(folder, images_folder)
                    self.rename_images(folder, f'{md_file[:-6]}.md', images_folder)


    def convert_to_markdown(self, folder: str = None, file: str = None):
        command = f'cmd /c jupyter nbconvert --output-dir=_{folder} --to markdown -- _notebooks/{folder}/{file}.ipynb'
        os.system(command)
        print('-- Converted all notebooks to markdown --')

    def relocate_images(self, folder, file_name: str):
        output_folder = f'images/{file_name}'

        # Create directory in images if it doesn't exist
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Gets all newest (In last hour) directories in _posts directory and
        """all_files_input_path = os.listdir('_posts')
        dirs = []
        for obj in os.listdir('_posts'):
            if '.md' not in obj:
                t = time.strptime(time.ctime(os.path.getmtime(f'_posts/{obj}')))
                if t[0] == self.current_time[0] and t[1] == self.current_time[1] and t[2] == self.current_time[2] and t[3] == self.current_time[3]:
                    dirs.append(obj)"""

        # Copies over everything from one directory in _posts to the directory in images

        for file in os.listdir(f'_{folder}/{file_name}'):
            source = f'_{folder}/{file_name}/{file}'
            destination = f'images/{file_name}/{file}'
            print(source, destination)
            if os.path.isfile(source):
                shutil.move(source, destination)
                print(f' --- Moved {source} to {destination} --- ')

        # Deletes empty directory
        os.rmdir(f'_{folder}/{file_name}')
        print('-- Relocated all notebook images to images directory --')

    def rename_images(self, folder: str, md_files: str, images_folder: str):
        content = [line for line in open(f'_{folder}/{md_files}')]
        writer = open(f'_{folder}/{md_files}', 'w')

        for index, line in enumerate(content):
            if '![png]' in line:
                new_line = f'![png](../images/{line[7:-2]})\n'
                content[index] = new_line

        writer.write(''.join(content))

        print('-- Renamed all images references in markdown files --')

if __name__ == '__main__':
    if sys.argv[1] == 'all':
        for folder in os.listdir('_notebooks'):
            FR = FileReformater(folder)
    else:
        FR = FileReformater(sys.argv[1])