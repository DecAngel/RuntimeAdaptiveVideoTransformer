from ravt.launchers import TensorboardLauncher


def main():
    server = TensorboardLauncher()
    server.run()


if __name__ == '__main__':
    main()
