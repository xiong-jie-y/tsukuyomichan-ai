from generate_talking_video import generate_video
import click

@click.command()
@click.argument('filename')
def main(filename):
    manuscript = "".join(open(filename).readlines())
    generate_video("test.mp4", manuscript, "subtitle.srt")

if __name__ == "__main__":
    main()