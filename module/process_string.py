def extract_tunnel_url():
    with open("tunnel.log") as file:
        for line in file:
            if "trycloudflare.com" in line:
                paths = line.strip().split()
                for i in paths:
                    if "trycloudflare.com" in i:
                        if "https://" in i:
                            return i